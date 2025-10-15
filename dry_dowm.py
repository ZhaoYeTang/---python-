# drydown_gleam_v42a_ROI.py
# -*- coding: utf-8 -*-
# 依赖：pip install xarray dask numpy pandas

'''
寻找干涸段

输入：gleam的SM数据（.nc）
输出：事件编号图、掩膜图、平滑后的土壤水分
'''

import numpy as np
import pandas as pd
import xarray as xr

# ========= 配置 =========
IN_NC  = r"E:\southWestern-dongmeng\SM\rawData\SMs_2024_GLEAM_v4.2a.nc"  # 你的 GLEAM v4.2a 日尺度文件
VAR    = "SMs"                                                  # 变量名（你文件里是 SMs）
OUT_NC = r"E:\southWestern-dongmeng\SM\GLEAMv42a_drydown_2024_ROI1.nc"

# 只处理的地理范围（北纬 31°~2°；东经 90°~111°）
LAT_MAX, LAT_MIN = 31.0, 2.0         # 注意：之后会统一为升序再裁剪
LON_MIN, LON_MAX = 90.0, 114.0

# 干涸段参数
SMOOTH_WIN = 5         # 日尺度中位数平滑窗口（3~7 都可）
EPS        = 5e-4      # 降幅容差（m3/m3/日），0.0002~0.001 常用
MIN_LEN    = 5         # 干涸段最短天数（天）
MAX_BREAK  = 1         # 允许的非下降小缺口天数
# ====================================

def label_drydown_events_1d(flags, min_len=5, max_break=1):
    """flags: 1D bool，True=下降或近似下降；返回 0/事件ID 的 int32 序列"""
    n = flags.shape[0]
    out = np.zeros(n, dtype=np.int32)
    i = 0; evt = 0
    while i < n:
        if not flags[i]:
            i += 1; continue
        start = i
        j = i
        while j < n:
            if flags[j]:
                j += 1
            else:
                k = j
                while k < n and (not flags[k]):
                    k += 1
                gap = k - j
                if gap <= max_break:
                    j = k
                else:
                    break
        end = j
        if end - start >= min_len:
            evt += 1
            out[start:end] = evt
        i = max(end, i+1)
    return out

def main():
    # 打开数据（惰性读取）
    ds = xr.open_dataset(IN_NC, chunks={"time": 120})
    da = ds[VAR].astype("float32")  # dims: time, lat, lon

    # 缺测处理（-999 → NaN；xarray 通常已处理，这里保险）
    da = da.where(da > -900)

    # 纬度升序，便于 slice（你的文件 lat 是递减的）
    if "lat" in da.dims and float(da.lat[1] - da.lat[0]) < 0:
        da = da.sortby("lat")

    # （可选）经度规范：GLEAM v4.2a 通常是 [-179.95, 179.95] 步长 0.1°
    # 你的 ROI (90~111E) 在两种经度体系（-180~180 或 0~360）都不跨经度 180°，无需改动。
    # 若以后遇到 0~360 的经度，可在这里把经度映射到 -180~180 再裁剪。

    # 空间裁剪（注意 lat 升序后，用 slice(LAT_MIN, LAT_MAX)）
    da_roi = da.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    # 平滑
    da_smooth = da_roi.rolling(
        time=SMOOTH_WIN, center=True, min_periods=max(3, SMOOTH_WIN//2)
    ).median()

    # 一阶差分与下降判据（含“近似不变”的容忍）
    dsm = da_smooth.diff("time")
    dsm_full = xr.full_like(da_smooth, np.nan)
    dsm_full.loc[dict(time=da_smooth.time[1:])] = dsm

    decreasing  = dsm_full <= -EPS
    almost_flat = np.abs(dsm_full) < EPS
    decr_or_flat = (decreasing | almost_flat).fillna(False)
    
    decr_or_flat = decr_or_flat.chunk({'time': -1})     # 单块化 time

    # 标注干涸事件 ID（逐像元）
    evt_id = xr.apply_ufunc(
        label_drydown_events_1d,
        decr_or_flat,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
        kwargs={"min_len": MIN_LEN, "max_break": MAX_BREAK}
    ).assign_coords(time=da_smooth.time)

    dry_mask = (evt_id > 0)

    # 质检：打印 ROI 范围、像元数、干涸覆盖率分布
    nlat = da_roi.sizes["lat"]; nlon = da_roi.sizes["lon"]
    print(f"ROI: lat {LAT_MIN}~{LAT_MAX} N, lon {LON_MIN}~{LON_MAX} E -> 网格 {nlat}×{nlon}")
    frac = dry_mask.mean("time", skipna=True).values
    q = np.nanpercentile(frac, [5,25,50,75,95])
    print("干涸掩膜年均覆盖率（%）五数概括：", (q*100).round(2).tolist())

    # 保存（仅 ROI）
    xr.Dataset({
        "SMs_smooth":        da_smooth,
        "drydown_event_id":  evt_id,
        "drydown_mask":      dry_mask
    }).to_netcdf(OUT_NC)
    print("已保存：", OUT_NC)

if __name__ == "__main__":
    main()
