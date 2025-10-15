# nc_to_tiff_qc.py
# -*- coding: utf-8 -*-
"""
把 NetCDF 的断点回归结果导出为 GeoTIFF：
- 输出：tau_ssm, slope_lo, slope_hi, delta_BIC, n_samples, keep_ratio
- 额外：tau_ssm_qc（按阈值做质控遮罩）
- CRS: EPSG:4326；压缩：LZW；NoData: -9999
"""

import os
import numpy as np
import xarray as xr

# ========= 路径与参数（改这里） =========
IN_NC   = r"E:\southWestern-dongmeng\breakpoint\GPP_SSM_breakpoint_2024_ROI.nc"  # 你刚生成的结果
OUT_DIR = r"E:\southWestern-dongmeng\breakpoint\2024"                      # 输出文件夹
VARS    = ["tau_ssm", "slope_lo", "slope_hi", "delta_BIC", "n_samples", "keep_ratio"]
FILL    = -9999.0
# 质控阈值（用于 tau_ssm_qc）
MIN_SAMPLES   = 12
MIN_DELTA_BIC = 10.0
# =====================================

def get_geotransform_from_latlon(lat, lon):
    """根据 1D lat/lon（中心点坐标）构造 GeoTransform（西北角为原点；north-up）"""
    lat = np.asarray(lat); lon = np.asarray(lon)
    # 分辨率（假设等间距）
    dx = float(np.abs(lon[1] - lon[0]))
    dy = float(np.abs(lat[1] - lat[0]))
    # 目标要求：数组的第 0 行是“北边” → 若当前 lat 升序（南→北），我们写之前要 flipud
    # GeoTransform 采用像元外框 → 左上角坐标：
    xmin = float(lon.min() - dx/2.0)
    ymax = float(lat.max() + dy/2.0)
    return xmin, dx, 0.0, ymax, 0.0, -dy  # (GT[0], GT[1], GT[2], GT[3], GT[4], GT[5])

def write_tiff_rasterio(path, arr2d, gt, nodata, dtype):
    import rasterio
    from rasterio.transform import Affine
    xmin, dx, _, ymax, _, neg_dy = gt
    transform = Affine.translation(xmin, ymax) * Affine.scale(dx, neg_dy)  # neg_dy为负
    profile = {
        "driver": "GTiff",
        "height": arr2d.shape[0],
        "width":  arr2d.shape[1],
        "count":  1,
        "dtype":  dtype,
        "crs":    "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "compress": "LZW",
        "predictor": 2 if np.issubdtype(np.dtype(dtype), np.floating) else 1,
        "nodata": nodata,
        "BIGTIFF": "IF_SAFER",
    }
    # 将 NaN 替换为 nodata
    out = np.where(np.isfinite(arr2d), arr2d, nodata).astype(dtype, copy=False)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

def write_tiff_gdal(path, arr2d, gt, nodata, dtype):
    from osgeo import gdal, osr
    gdal.UseExceptions()
    dtype_map = {
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
        "int16":   gdal.GDT_Int16,
        "int32":   gdal.GDT_Int32,
        "uint8":   gdal.GDT_Byte,
    }
    gdal_dtype = dtype_map.get(str(np.dtype(dtype)))
    if gdal_dtype is None:
        gdal_dtype = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    ny, nx = arr2d.shape
    ds = driver.Create(path, nx, ny, 1, gdal_dtype,
                       options=["TILED=YES","COMPRESS=LZW","PREDICTOR=2","BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    out = np.where(np.isfinite(arr2d), arr2d, nodata).astype(dtype, copy=False)
    band.WriteArray(out)
    band.FlushCache()
    ds.FlushCache()
    ds = None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = xr.open_dataset(IN_NC, decode_times=True, mask_and_scale=True)

    # 基础检查
    for v in VARS:
        if v not in ds:
            raise KeyError(f"变量 {v} 不在 {IN_NC} 中。可用变量：{list(ds.data_vars)}")
    lat = ds["lat"].values
    lon = ds["lon"].values
    # 计算 geotransform；并把数组翻到“北在上”
    gt = get_geotransform_from_latlon(lat, lon)
    flip_needed = True  # 你的 lat 通常是升序（南→北），必须 flipud 才是 north-up
    # 选择写驱动
    try:
        import rasterio  # noqa
        use_rasterio = True
    except Exception:
        use_rasterio = False

    def _write_one(varname, dtype="float32"):
        arr = ds[varname].values  # (lat, lon)
        if flip_needed:
            arr = np.flipud(arr)
        out_path = os.path.join(OUT_DIR, f"{varname}.tif")
        if use_rasterio:
            write_tiff_rasterio(out_path, arr, gt, FILL, dtype)
        else:
            write_tiff_gdal(out_path, arr, gt, FILL, dtype)
        print("写出：", out_path)

    # 主变量
    _write_one("tau_ssm",    dtype="float32")
    _write_one("slope_lo",   dtype="float32")
    _write_one("slope_hi",   dtype="float32")
    _write_one("delta_BIC",  dtype="float32")
    # n_samples 可能是整型
    ns = ds["n_samples"].astype("int16").values
    if flip_needed: ns = np.flipud(ns)
    out_ns = os.path.join(OUT_DIR, "n_samples.tif")
    if use_rasterio:
        write_tiff_rasterio(out_ns, ns, gt, -32768, "int16")
    else:
        write_tiff_gdal(out_ns, ns, gt, -32768, "int16")
    print("写出：", out_ns)
    # keep_ratio
    _write_one("keep_ratio", dtype="float32")

    # 质控版 tau（可调 MIN_SAMPLES、MIN_DELTA_BIC）
    good = (ds["n_samples"] >= MIN_SAMPLES) & (ds["delta_BIC"] > MIN_DELTA_BIC)
    tau_qc = ds["tau_ssm"].where(good).values
    if flip_needed:
        tau_qc = np.flipud(tau_qc)
    out_qc = os.path.join(OUT_DIR, f"tau_ssm_qc_ns{MIN_SAMPLES}_dbic{int(MIN_DELTA_BIC)}.tif")
    if use_rasterio:
        write_tiff_rasterio(out_qc, tau_qc, gt, FILL, "float32")
    else:
        write_tiff_gdal(out_qc, tau_qc, gt, FILL, "float32")
    print("写出：", out_qc)

    # 可选：为所有浮点图层写一个 0–1 有效性蒙版（方便 GIS 里快速查看）
    valid = np.isfinite(ds["tau_ssm"].values).astype("uint8")
    if flip_needed: valid = np.flipud(valid)
    out_valid = os.path.join(OUT_DIR, "valid_mask.tif")
    if use_rasterio:
        write_tiff_rasterio(out_valid, valid, gt, 0, "uint8")
    else:
        write_tiff_gdal(out_valid, valid, gt, 0, "uint8")
    print("写出：", out_valid)

if __name__ == "__main__":
    main()
