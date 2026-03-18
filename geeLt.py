# -*- coding: utf-8 -*-
"""
GEE LandTrendr segmentation export (strictly GEE LandTrendr) + robust task start
- Exports ONLY segmentation info (sy/ey/sv/ev for 6 segments) per tile to Drive
- Fixes:
  1) arrayPad signature issue (Python API): use arrayCat + arraySlice instead
  2) band name issue "1.0": use format('%d') to avoid dots in band names
  3) request_id conflict at large task counts: force unique request_id (uuid)
  4) resume support: start from RESUME_FROM to avoid restarting earlier tiles

@author: Tangyi
"""

import time
import uuid
import ee


# -------------------------
# 0) EE init
# -------------------------
PROJECT_ID = "forest-480203"  # your Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

print("EE initialized with project:", PROJECT_ID)


# -------------------------
# 1) Parameters
# -------------------------
ASSET_TRAIN_POLYGONS = "projects/forest-480203/assets/train_polygons"
DRIVE_FOLDER = "GEE_export"
ROI = ee.Geometry.Rectangle([92.0, 5.0, 113.5, 29.5])

START_YEAR, END_YEAR = 2004, 2024
BASE_YEAR = 2014
N_TREES = 200

TILE_DEG = 0.5
SCALE = 500

POLY_SUBSAMPLE = 0.33
SAMP_SUBSAMPLE = 0.60

# LandTrendr params (keep identical to your JS)
LT_MAX_SEG = 6
LT_PARAMS = dict(
    maxSegments=LT_MAX_SEG,
    spikeThreshold=0.9,
    vertexCountOvershoot=3,
    preventOneYearRecovery=True,
    recoveryThreshold=0.25,
    pvalThreshold=0.05,
    bestModelProportion=0.75,
    minObservationsNeeded=6
)

# Task throttling
START_BATCH = 20
SLEEP_BETWEEN_BATCH_SEC = 5

'''
在这里修改开始与结束的tile编号
'''
# Resume control 
RESUME_FROM = 154   # set to 0 for a fresh run;

# Optionally cap submission range to avoid too many queued tasks
END_AT = 155  # e.g. 1700; or None to go to the end


# -------------------------
# 2) Data prep: annual MCD43A4 + topo + RF forest_prob
# -------------------------
REfl_BANDS = [
    "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3",
    "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band5", "Nadir_Reflectance_Band6",
    "Nadir_Reflectance_Band7"
]
FEATURE_BANDS = REfl_BANDS + ["dem", "slope", "asp_sin", "asp_cos"]

def scale_mcd43(img):
    return ee.Image(img).select(REfl_BANDS).multiply(0.0001).copyProperties(img, ["system:time_start"])

mcd43 = (ee.ImageCollection("MODIS/061/MCD43A4")
         .filterBounds(ROI)
         .filterDate(f"{START_YEAR}-01-01", f"{END_YEAR+1}-01-01")
         .map(scale_mcd43))

years = ee.List.sequence(START_YEAR, END_YEAR)

def make_annual(y):
    y = ee.Number(y)
    return (mcd43.filter(ee.Filter.calendarRange(y, y, "year"))
            .median()
            .select(REfl_BANDS)
            .clip(ROI)
            .set("year", y))

annual = ee.ImageCollection(years.map(make_annual))

# topo: SRTM + slope + aspect sin/cos (reproject to MODIS 500m)
dem30 = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(ROI)
terrain = ee.Terrain.products(dem30)
slope30 = terrain.select("slope")
aspect30 = terrain.select("aspect")
aspect_rad = aspect30.multiply(3.141592653589793 / 180.0)
asp_sin30 = aspect_rad.sin().rename("asp_sin")
asp_cos30 = aspect_rad.cos().rename("asp_cos")

modis_proj = ee.Image(annual.first()).projection()
dem500 = dem30.resample("bilinear").reproject(crs=modis_proj, scale=500).rename("dem")
slope500 = slope30.resample("bilinear").reproject(crs=modis_proj, scale=500).rename("slope")
asp_sin500 = asp_sin30.resample("bilinear").reproject(crs=modis_proj, scale=500)
asp_cos500 = asp_cos30.resample("bilinear").reproject(crs=modis_proj, scale=500)
topo500 = ee.Image.cat([dem500, slope500, asp_sin500, asp_cos500])

annual_topo = annual.map(lambda img: ee.Image(img).addBands(topo500).copyProperties(img, ["year"]))

# training points
train_fc = ee.FeatureCollection(ASSET_TRAIN_POLYGONS)
train_fc = train_fc.filter(ee.Filter.inList("class", [0, 1]))
train_fc = train_fc.randomColumn("rp").filter(ee.Filter.lt("rp", POLY_SUBSAMPLE))

def poly_to_point(f):
    f = ee.Feature(f)
    return f.simplify(1000).centroid(10).copyProperties(f)

train_pts = train_fc.map(poly_to_point)

# sample training table
base_img = annual_topo.filter(ee.Filter.eq("year", BASE_YEAR)).first()
samples_all = base_img.select(FEATURE_BANDS).sampleRegions(
    collection=train_pts,
    properties=["class"],
    scale=500,
    geometries=False,
    tileScale=2
)
samples = samples_all.randomColumn("rs").filter(ee.Filter.lt("rs", SAMP_SUBSAMPLE))

# RF classifier + annual forest_prob
rf = ee.Classifier.smileRandomForest(numberOfTrees=N_TREES).train(
    features=samples,
    classProperty="class",
    inputProperties=FEATURE_BANDS
)
rf_prob = rf.setOutputMode("PROBABILITY")

def add_prob(img):
    img = ee.Image(img)
    p = img.select(FEATURE_BANDS).classify(rf_prob).rename("forest_prob")
    return img.addBands(p).copyProperties(img, ["year"])

annual_prob = annual_topo.map(add_prob)


# -------------------------
# 3) LandTrendr segmentation export per tile (sy/ey/sv/ev, fixed 6 seg)
# -------------------------
def lt_segments_for_region(region_geom):
    def to_ts(img):
        img = ee.Image(img)
        y = ee.Number(img.get("year"))
        return (img.select("forest_prob")
                .clamp(0, 1)
                .rename("fp")
                .set("system:time_start", ee.Date.fromYMD(y, 1, 1).millis())
                .clip(region_geom))

    ts = annual_prob.sort("year").map(to_ts)

    lt = ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=ts, **LT_PARAMS)
    arr = ee.Image(lt).select("LandTrendr")

    vflag = arr.arraySlice(0, 3, 4)     # vertexFlag row
    verts = arr.arrayMask(vflag)        # keep vertices only

    vyear = verts.arraySlice(0, 0, 1)   # [1, nVert]
    vfit  = verts.arraySlice(0, 2, 3)   # [1, nVert]

    # segments from adjacent vertices
    sy = vyear.arraySlice(1, 0, -1)     # [1, nSeg]
    ey = vyear.arraySlice(1, 1, None)
    sv = vfit.arraySlice(1, 0, -1)
    ev = vfit.arraySlice(1, 1, None)

    # ---- fixlen to exactly 6 segments: arrayCat zeros then slice ----
    zeros = ee.Image.constant(0).toArray().arrayRepeat(1, LT_MAX_SEG)  # [1, 6]

    def fixlen(x):
        x = ee.Image(x)
        x2 = x.arrayCat(zeros, 1)                  # [1, nSeg+6]
        return x2.arraySlice(1, 0, LT_MAX_SEG)     # [1, 6]

    sy6 = fixlen(sy)
    ey6 = fixlen(ey)
    sv6 = fixlen(sv)
    ev6 = fixlen(ev)

    # band names: seg_1..seg_6 (avoid "1.0" which introduces '.')
    seg_names = ee.List.sequence(1, LT_MAX_SEG).map(
        lambda k: ee.String("seg_").cat(ee.Number(k).format('%d'))
    )

    sy_img = sy6.arrayFlatten([["sy"], seg_names])
    ey_img = ey6.arrayFlatten([["ey"], seg_names])
    sv_img = sv6.arrayFlatten([["sv"], seg_names])
    ev_img = ev6.arrayFlatten([["ev"], seg_names])

    out = ee.Image.cat([sy_img, ey_img, sv_img, ev_img]).toFloat()
    return out.clip(region_geom)


# -------------------------
# 4) Build tiles (0.5 degree) and start export tasks with unique request_id
# -------------------------
lonlat = ee.Image.pixelLonLat()
lon_grid = lonlat.select("longitude").divide(TILE_DEG).floor()
lat_grid = lonlat.select("latitude").divide(TILE_DEG).floor()

# keep your original tile_id rule (note: collisions are unlikely here, but request_id is fixed anyway)
tile_id = lon_grid.multiply(10000).add(lat_grid).toInt().rename("tile")

tiles = tile_id.reduceToVectors(
    geometry=ROI,
    scale=20000,
    geometryType="polygon",
    labelProperty="tile",
    reducer=ee.Reducer.countEvery()
)

tiles_list = tiles.toList(tiles.size())
n = tiles.size().getInfo()
print("Tile count =", n)

start_i = int(RESUME_FROM)
end_i = int(END_AT) if END_AT is not None else n

print(f"Submitting tasks for tiles [{start_i}, {end_i})")

submitted = 0
for i in range(start_i, end_i):
    geom = ee.Feature(tiles_list.get(i)).geometry()
    img = lt_segments_for_region(geom)

    desc = f"LT_segments_fp_{START_YEAR}_{END_YEAR}_tile_{i}"

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=DRIVE_FOLDER,
        fileNamePrefix=desc,
        region=geom,
        scale=SCALE,
        maxPixels=1e13
    )

    # ✅ critical fix: force unique request_id to avoid 400 conflicts after reruns
    task._request_id = f"lt-{i}-{uuid.uuid4().hex[:10]}"

    try:
        task.start()
        submitted += 1
        print(f"Started {i+1}/{n}: {desc}")
    except Exception as e:
        print(f"[FAIL] tile {i} ({desc}): {e}")
        # continue to next tile
        continue

    if (submitted % START_BATCH) == 0:
        time.sleep(SLEEP_BETWEEN_BATCH_SEC)

print("All segment-export tasks started (or attempted).")