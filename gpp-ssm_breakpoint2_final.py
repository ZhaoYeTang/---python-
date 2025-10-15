# gpp_ssm_breakpoint_tiles_2006.py
# -*- coding: utf-8 -*-
"""
GLASS GPP (8天, HDF, 多瓦片) × GLEAM SSM (日, 0.1°, ROI) → 单断点回归阈值 τ
要点：
- 先在内存里用 GDAL BuildVRT + Warp 把同一天的瓦片拼起来并重投影到 EPSG:4326
- Warp 时强制 Float32 + dstNodata=-9999，避免 nodata 被“夹成 0”的问题
- 只在“干涸窗口 && 当天有 GPP”的像元上做回归
依赖（conda-forge）：gdal xarray dask numpy pandas netcdf4

优化后的，寻找水分限制阈值（断点回归）
"""

import os, re, datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import dask
from osgeo import gdal

# ======= 你的输入 / 输出 =======
# 已做“干涸识别”的 SSM（含 SMs_smooth & drydown_mask）
SM_NC      = r"E:\southWestern-dongmeng\SM\GLEAMv42a_drydown_2005_ROI.nc"
VAR_SMOOTH = "SMs_smooth"
VAR_MASK   = "drydown_mask"

# YYYY 年 GLASS GPP 瓦片目录（文件名形如 GLASS12E01.V60.A2006241.h27v07.2005059.hdf）
GPP_DIR    = r"E:\southWestern-dongmeng\GPP\2005"
FILE_REGEX = r"^GLASS12E01\.V60\.A2005\d{3}\.h\d{2}v\d{2}\.\d+\.hdf$"  # AYYYYDDD & hXXvYY   这里要改年份

# 输出
OUT_NC     = r"E:\southWestern-dongmeng\breakpoint\GPP_SSM_breakpoint_2005_ROI.nc"
# =================================

# 窗口与筛选
WINDOW_LEFT, WINDOW_RIGHT = 4, 3     # [t-4, t+3]
P_THR        = 0.80                   # 干涸覆盖率阈值
MIN_SAMPLES  = 12                     # 每像元最少样本
N_TAU        = 16                     # τ 候选扫描数（10~90 分位上等距取 N_TAU 个）
CH_LAT, CH_LON = 120, 120             # 计算时的空间分块

# ----------------- 工具函数 -----------------
def parse_time_from_name(fn: str) -> pd.Timestamp | None:
    """优先解析 AYYYYDDD；若存在 YYYYMMDD 也能识别。输出 8 天期中心日"""
    b = os.path.basename(fn)
    m = re.search(r'A(\d{4})(\d{3})', b)
    if m:
        y, doy = int(m.group(1)), int(m.group(2))
        t0 = pd.Timestamp(dt.date(y, 1, 1)) + pd.Timedelta(days=doy - 1)
        return t0 + pd.Timedelta(days=3)
    m = re.search(r'(\d{8})', b)
    if m:
        y = int(b[m.start():m.start()+4]); mo = int(b[m.start()+4:m.start()+6]); d = int(b[m.start()+6:m.start()+8])
        t0 = pd.Timestamp(dt.date(y, mo, d))
        return t0 + pd.Timedelta(days=3)
    return None

def pick_gpp_subdataset(hdf_path: str) -> str:
    """挑出 GPP 的子数据集（支持 HDF4/HDF5）"""
    ds = gdal.Open(hdf_path)
    subs = ds.GetSubDatasets()
    if not subs:
        return hdf_path
    for name, desc in subs:
        if 'gpp' in (name.lower() + desc.lower()):
            return name
    return subs[0][0]

def mosaic_tiles_to_roi(tile_paths, target_lat, target_lon) -> xr.DataArray:
    """
    把同一天的瓦片拼接成一张图，并 Warp 到 SSM ROI 网格：
    - 强制 Float32 + dstNodata=-9999（防止 nodata 被夹成 0）
    - Warp 后应用 band 的 scale/offset（若存在）
    - 与 SSM 网格最近邻对齐（消浮点边界误差）
    """
    # 1) 选出每块中的 GPP 子数据集
    sd_list = [pick_gpp_subdataset(p) for p in tile_paths]

    # 2) 尝试读取源 nodata（若有）
    src_nodata = None
    try:
        s0 = gdal.Open(sd_list[0])
        src_nodata = s0.GetRasterBand(1).GetNoDataValue()
    except Exception:
        pass

    # 3) 生成 VRT（内存）
    vrt_path = "/vsimem/mosaic.vrt"
    gdal.BuildVRT(
        vrt_path, sd_list,
        srcNodata=src_nodata if src_nodata is not None else None,
        VRTNodata=src_nodata if src_nodata is not None else None
    )

    # 4) 依据 SSM 网格计算 ROI 外包框（像元外框）
    lat = np.asarray(target_lat); lon = np.asarray(target_lon)
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    xmin = float(lon.min() - dlon / 2); xmax = float(lon.max() + dlon / 2)
    ymin = float(lat.min() - dlat / 2); ymax = float(lat.max() + dlat / 2)

    # 5) Warp 到 ROI：关键参数 outputType=Float32, dstNodata=-9999
    dst_path = "/vsimem/roi.tif"
    warp_opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        xRes=dlon, yRes=dlat,
        outputBounds=(xmin, ymin, xmax, ymax),
        resampleAlg="bilinear",
        srcNodata=src_nodata,
        dstNodata=-9999,
        outputType=gdal.GDT_Float32,
        multithread=True
    )
    gdal.Warp(dst_path, vrt_path, options=warp_opts)

    # 6) 读回数组，应用 scale/offset，并转 NaN
    dst = gdal.Open(dst_path)
    band = dst.GetRasterBand(1)
    arr = band.ReadAsArray().astype("f4")
    # 应用 scale/offset（若有）
    scale = band.GetScale(); offset = band.GetOffset()
    if scale not in (None, 1.0) or (offset not in (None, 0.0)):
        if scale is None:  scale = 1.0
        if offset is None: offset = 0.0
        arr = arr * float(scale) + float(offset)
    # nodata -> NaN；并把负值（物理不合理）置 NaN
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(arr < 0, np.nan, arr)

    # 7) 生成 1D 坐标（确保纬度升序）
    gt = dst.GetGeoTransform()  # (xmin, dx, 0, ymax, 0, -dy)
    nx, ny = dst.RasterXSize, dst.RasterYSize
    dx, dy = gt[1], abs(gt[5])
    lon_out = gt[0] + dx * (np.arange(nx) + 0.5)
    lat_out = gt[3] - dy * (np.arange(ny) + 0.5)
    if lat_out[1] < lat_out[0]:
        arr = arr[::-1, :]
        lat_out = lat_out[::-1]

    da = xr.DataArray(
        arr, coords={"lat": lat_out.astype("f4"), "lon": lon_out.astype("f4")},
        dims=("lat", "lon"), name="GPP"
    )

    # 8) 最近邻对齐到 SSM 网格（避免边界微差）
    da = da.interp(lat=target_lat, lon=target_lon, method="nearest")

    # 9) 清理内存文件
    dst = None
    gdal.Unlink(vrt_path); gdal.Unlink(dst_path)
    return da

def fit_breakpoint_1d(x, y, min_samples=12, n_tau=16):
    """单断点分段线性：y = a + b1*x + b2*max(0,x-τ)，扫描 τ 最小 SSE，返回 (τ, 下段斜率, 上段斜率, ΔBIC, n)"""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < min_samples or np.nanstd(x) <= 1e-9:
        return np.nan, np.nan, np.nan, np.nan, n
    # 单线
    X1 = np.column_stack([np.ones(n), x])
    beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat1 = X1 @ beta1
    sse1 = np.sum((y - yhat1)**2)

    lo = np.nanpercentile(x, 10); hi = np.nanpercentile(x, 90)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan, np.nan, n
    taus = np.linspace(lo, hi, int(n_tau))

    best_sse = np.inf; best_tau = np.nan; best_s1 = np.nan; best_s2 = np.nan
    for tau in taus:
        h = np.maximum(0.0, x - tau)
        X = np.column_stack([np.ones(n), x, h])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        sse = np.sum((y - yhat)**2)
        if sse < best_sse:
            best_sse = sse; best_tau = tau
            b1, b2 = beta[1], beta[2]
            best_s1 = b1
            best_s2 = b1 + b2

    # ΔBIC：单线(2参) vs 断点(3参)
    k1, k2 = 2, 3
    delta_bic = (np.nan if best_sse <= 0 or sse1 <= 0
                 else (n*np.log(sse1/n) + k1*np.log(n)) - (n*np.log(best_sse/n) + k2*np.log(n)))
    return best_tau, best_s1, best_s2, delta_bic, n

# ----------------- 主流程 -----------------
def main():
    # 0) 读 SSM（已干涸）
    ds = xr.open_dataset(SM_NC, decode_times=True, mask_and_scale=True)
    if VAR_SMOOTH not in ds or VAR_MASK not in ds:
        raise KeyError(f"{SM_NC} 中缺少 {VAR_SMOOTH}/{VAR_MASK}")
    ssm = ds[VAR_SMOOTH].astype("f4")
    dry = ds[VAR_MASK].astype(bool)
    if ssm.lat.size > 1 and float(ssm.lat[1] - ssm.lat[0]) < 0:
        ssm = ssm.sortby("lat"); dry = dry.sortby("lat")
    target_lat, target_lon = ssm.lat, ssm.lon

    # 1) 按日期收集瓦片
    files = [os.path.join(GPP_DIR, f) for f in os.listdir(GPP_DIR) if re.match(FILE_REGEX, f, re.I)]
    groups = {}
    for fp in files:
        t = parse_time_from_name(fp)
        if t is None:
            print(f"[跳过] 无法解析日期：{os.path.basename(fp)}"); continue
        groups.setdefault(pd.Timestamp(t).normalize(), []).append(fp)
    if not groups:
        raise RuntimeError("未找到可解析日期的 GPP 瓦片。")

    # 2) 拼接→裁剪→对齐
    gpp_list = []
    for day, tpaths in sorted(groups.items()):
        try:
            da = mosaic_tiles_to_roi(tpaths, target_lat, target_lon)
        except Exception as e:
            print(f"[跳过 {day.date()}] 瓦片拼接失败：{e}")
            continue
        da = da.where(np.isfinite(da) & (da >= 0))
        t_center = day + pd.Timedelta(days=3)  # 8天期中心日
        gpp_list.append(da.expand_dims(time=[t_center]))

    if not gpp_list:
        raise RuntimeError("拼接后没有可用 GPP 切片。")
    gpp = xr.concat(gpp_list, dim="time").sortby("time")
    # 去重
    gpp = gpp.sel(time=~gpp.get_index('time').duplicated())

    # 3) 8天窗口与干涸交集（并强制要求“当天有 GPP”）
    ssm8_list, keep_list = [], []
    for t in gpp.time.values:
        t = pd.Timestamp(t)
        w0, w1 = t - pd.Timedelta(days=WINDOW_LEFT), t + pd.Timedelta(days=WINDOW_RIGHT)
        dry_win = dry.sel(time=slice(w0, w1))
        ssm_win = ssm.sel(time=slice(w0, w1))
        num = dry_win.sum('time', skipna=True); den = dry_win.count('time')
        p = xr.where(den > 0, num / den, 0.0)
        keep = (p >= P_THR) & gpp.sel(time=t).notnull()  # GPP 覆盖必须有
        ssm8 = ssm_win.median('time', skipna=True).where(keep)
        ssm8_list.append(ssm8.expand_dims(time=[t]))
        keep_list.append(keep.expand_dims(time=[t]))

    ssm_use  = xr.concat(ssm8_list, dim='time').sortby('time')
    keepmask = xr.concat(keep_list, dim='time').sortby('time')
    gpp_use  = gpp.where(keepmask)

    # 4) 逐像元断点回归（dask 并行；time 为单块）
    ssm_use = ssm_use.chunk({'time': -1, 'lat': CH_LAT, 'lon': CH_LON})
    gpp_use = gpp_use.chunk({'time': -1, 'lat': CH_LAT, 'lon': CH_LON})

    def comp_scalar(x, y, idx):
        tau, s1, s2, dbic, n = fit_breakpoint_1d(x, y, min_samples=MIN_SAMPLES, n_tau=N_TAU)
        return np.float64([tau, s1, s2, dbic, n][idx])

    with dask.config.set(scheduler="threads"):
        tau_map   = xr.apply_ufunc(comp_scalar, ssm_use, gpp_use, kwargs={'idx': 0},
                                   input_core_dims=[['time'], ['time']], dask='parallelized',
                                   output_dtypes=[np.float64], vectorize=True)
        slope_lo  = xr.apply_ufunc(comp_scalar, ssm_use, gpp_use, kwargs={'idx': 1},
                                   input_core_dims=[['time'], ['time']], dask='parallelized',
                                   output_dtypes=[np.float64], vectorize=True)
        slope_hi  = xr.apply_ufunc(comp_scalar, ssm_use, gpp_use, kwargs={'idx': 2},
                                   input_core_dims=[['time'], ['time']], dask='parallelized',
                                   output_dtypes=[np.float64], vectorize=True)
        delta_bic = xr.apply_ufunc(comp_scalar, ssm_use, gpp_use, kwargs={'idx': 3},
                                   input_core_dims=[['time'], ['time']], dask='parallelized',
                                   output_dtypes=[np.float64], vectorize=True)
        n_eff     = xr.apply_ufunc(comp_scalar, ssm_use, gpp_use, kwargs={'idx': 4},
                                   input_core_dims=[['time'], ['time']], dask='parallelized',
                                   output_dtypes=[np.float64], vectorize=True)
        tau_map, slope_lo, slope_hi, delta_bic, n_eff = dask.compute(
            tau_map, slope_lo, slope_hi, delta_bic, n_eff
        )

    # 5) 质量指标 & 输出
    gpp_valid_ratio = xr.where(np.isfinite(gpp), 1, 0).mean("time").astype("f4")
    ds_out = xr.Dataset({
        "tau_ssm":         tau_map.astype('f4'),
        "slope_lo":        slope_lo.astype('f4'),
        "slope_hi":        slope_hi.astype('f4'),
        "delta_BIC":       delta_bic.astype('f4'),
        "n_samples":       n_eff.astype('i2'),
        "keep_ratio":      keepmask.mean('time').astype('f4'),
        "gpp_valid_ratio": gpp_valid_ratio,
    }, coords={"lat": ssm.lat, "lon": ssm.lon})

    ds_out.attrs["notes"] = (
        f"GLASS 8d GPP（瓦片拼接，Float32+dstNodata=-9999）× GLEAM SSM（干涸窗口）；"
        f"窗口=[-{WINDOW_LEFT},+{WINDOW_RIGHT}]，p≥{P_THR}；MIN_SAMPLES={MIN_SAMPLES}；N_TAU={N_TAU}。"
    )
    ds_out.to_netcdf(OUT_NC)
    print("✅ 完成：", OUT_NC)

if __name__ == "__main__":
    main()
