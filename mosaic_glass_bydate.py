#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mosaic GLASS GPP HDF tiles by date and export GeoTIFFs
------------------------------------------------------
逻辑遵循你的断点回归脚本里的“瓦片拼接 + Warp 重投影 EPSG:4326、Float32、dstNodata=-9999”的思路，
但这里只做“同日多瓦片镶嵌 → 可选重投影 → 输出 GeoTIFF”，不做与 SSM 的对齐和后续分析。

依赖（conda-forge）：gdal numpy pandas
    conda install -c conda-forge gdal numpy pandas

用法：
  1) 修改 CONFIG 里的路径/年份/文件名正则；
  2) python mosaic_glass_bydate.py
输出：每个日期一张 GeoTIFF，文件名如：GPP_A2011001_mosaic.tif 或 GPP_20110104_mosaic.tif

注意：
- HDF 若“无子数据集（SDS）”，会直接把 HDF 当作一个栅格读取；
- 若存在 SDS，会优先挑选名字或描述里包含 "gpp" 的子数据集；
- 若源数据没有 CRS，可在 CONFIG 里设置 FORCE_SRC_CRS（例如 MODIS Sinusoidal: "SR-ORG:6974" 或其 PROJ 字符串）。
"""

import os
import re
import sys
import traceback
import datetime as dt
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

# -------------------- 配置 --------------------
CONFIG = dict(
    # 某一年的 GLASS GPP 瓦片目录
    GPP_DIR = r"E:\southWestern-dongmeng\GPP\2011",
    # 过滤文件名（示例：GLASS12E01.V60.A2011DDD.hXXvYY.yyyyddd.hdf）
    FILE_REGEX = r"^GLASS12E01\.V60\.A2011\d{3}\.h\d{2}v\d{2}\.\d+\.hdf$",
    # 输出目录
    OUT_DIR = r"E:\southWestern-dongmeng\GPP\mosaic\2011",

    # 可选：强制源 CRS（当 HDF/SDS 缺少投影信息时）
    # 例：MODIS Sinusoidal -> "SR-ORG:6974" 或 "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs"
    FORCE_SRC_CRS = None,  # e.g., "SR-ORG:6974"

    # 可选：是否重投影到 EPSG:4326；若 False 则保持原投影
    WARP_TO_EPSG4326 = True,

    # 可选：目标分辨率（度）；None 表示让 GDAL 自行选择
    # 若你清楚应为 0.05°，可设 xRes=0.05, yRes=0.05；否则建议 None
    XRES = None,
    YRES = None,

    # 其他 Warp/输出参数
    DST_NODATA = -9999.0,
    RESAMPLE = "bilinear",    # "nearest","bilinear","cubic"...
    COMPRESS = "DEFLATE",
    BIGTIFF = "YES",
    TILED = "YES",
)

# -------------------- 工具函数 --------------------
def parse_time_from_name(fn: str) -> tuple[pd.Timestamp | None, str | None]:
    """
    从文件名解析日期，优先 AYYYYDDD；若存在 YYYYMMDD 也能识别。
    返回 (中心日时间戳, 用于命名的日期token)，若均失败返回 (None, None)
    """
    b = os.path.basename(fn)
    m = re.search(r"A(\d{4})(\d{3})", b)
    if m:
        y, doy = int(m.group(1)), int(m.group(2))
        t0 = pd.Timestamp(dt.date(y, 1, 1)) + pd.Timedelta(days=doy - 1)
        center = t0 + pd.Timedelta(days=3)  # 8日中心
        return center, f"A{y}{doy:03d}"
    m2 = re.search(r"(\d{8})", b)
    if m2:
        s = m2.group(1)
        try:
            t = pd.to_datetime(s, format="%Y%m%d")
            return t, s
        except Exception:
            pass
    return None, None


def pick_gpp_subdataset(hdf_path: str) -> str:
    """
    优先挑出 GPP 的子数据集（支持 HDF4/HDF5）；
    若 HDF 无 SDS，则直接返回 hdf_path 本身。
    """
    ds = gdal.Open(hdf_path)
    subs = ds.GetSubDatasets()
    if not subs:
        return hdf_path
    # 优先包含 gpp 的
    for name, desc in subs:
        text = (name + " " + desc).lower()
        if "gpp" in text:
            return name
    # 否则 fallback: 第一个
    return subs[0][0]


def mosaic_one_day(sd_list: list[str],
                   out_tif: str,
                   force_src_crs: str | None = None,
                   warp_to_epsg4326: bool = True,
                   xres: float | None = None,
                   yres: float | None = None,
                   dst_nodata: float = -9999.0,
                   resample: str = "bilinear",
                   compress: str = "DEFLATE",
                   bigtiff: str = "YES",
                   tiled: str = "YES") -> None:
    """
    用 GDAL BuildVRT + Warp 把同日的 SDS 或 HDF 栅格镶嵌成一张图，并输出为 GeoTIFF。
    - 若 force_src_crs 不为空，会在 Warp 时以此作为 srcSRS。
    - 若 warp_to_epsg4326=True，则重投影到 EPSG:4326；否则输出保持源投影。
    - 若 xres/yres 为 None，让 GDAL 自选分辨率；否则采用指定分辨率（度）。
    - 输出强制 Float32 + dstNodata。
    - 会应用 band 的 scale/offset（若存在），并把负值置为 nodata。
    """
    if len(sd_list) == 0:
        raise RuntimeError("空 SDS 列表。")

    # 1) BuildVRT（在内存 /vsimem 上）
    vrt_path = "/vsimem/mosaic.vrt"
    # 尝试读取源 nodata
    src_nodata = None
    try:
        s0 = gdal.Open(sd_list[0])
        src_band = s0.GetRasterBand(1)
        src_nodata = src_band.GetNoDataValue()
    except Exception:
        pass

    gdal.BuildVRT(
        vrt_path, sd_list,
        srcNodata=src_nodata if src_nodata is not None else None,
        VRTNodata=src_nodata if src_nodata is not None else None
    )

    # 2) Warp 到目标
    tmp_tif = "/vsimem/mosaic_tmp.tif"
    warp_kwargs = dict(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        dstNodata=dst_nodata,
        multithread=True,
        resampleAlg=resample,
        creationOptions=[
            f"COMPRESS={compress}",
            f"TILED={tiled}",
            f"BIGTIFF={bigtiff}",
            "PREDICTOR=2"
        ]
    )
    if warp_to_epsg4326:
        warp_kwargs["dstSRS"] = "EPSG:4326"
        if xres is not None and yres is not None:
            warp_kwargs["xRes"] = float(xres)
            warp_kwargs["yRes"] = float(yres)

    if force_src_crs:
        warp_kwargs["srcSRS"] = force_src_crs

    gdal.Warp(tmp_tif, vrt_path, **warp_kwargs)

    # 3) 读回数组，应用 scale/offset 与负值屏蔽，然后重新写成最终 GeoTIFF
    dst = gdal.Open(tmp_tif)
    band = dst.GetRasterBand(1)
    arr = band.ReadAsArray().astype("f4")

    scale = band.GetScale(); offset = band.GetOffset()
    if (scale is not None and float(scale) != 1.0) or (offset is not None and float(offset) != 0.0):
        arr = arr * (1.0 if scale is None else float(scale)) + (0.0 if offset is None else float(offset))

    # 负值 → nodata
    arr = np.where(arr < 0, dst_nodata, arr)

    # 写出
    gt = dst.GetGeoTransform()
    proj = dst.GetProjection()
    nx, ny = dst.RasterXSize, dst.RasterYSize

    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_tif, nx, ny, 1, gdal.GDT_Float32,
        options=[
            f"COMPRESS={compress}",
            f"TILED={tiled}",
            f"BIGTIFF={bigtiff}",
            "PREDICTOR=2"
        ]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)
    out_band.SetNoDataValue(dst_nodata)
    out_band.FlushCache()

    # 清理
    out_band = None; out_ds = None
    dst = None
    gdal.Unlink(vrt_path); gdal.Unlink(tmp_tif)


# -------------------- 主流程 --------------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    # 收集并按“日期 token”分组
    files = [os.path.join(cfg["GPP_DIR"], f) for f in os.listdir(cfg["GPP_DIR"]) if re.match(cfg["FILE_REGEX"], f, re.I)]
    if not files:
        print(f"[ERROR] 目录为空或正则不匹配：{cfg['GPP_DIR']}")
        sys.exit(2)

    groups = {}  # token -> list of file paths
    centers = {} # token -> center timestamp
    for fp in files:
        center, token = parse_time_from_name(fp)
        if token is None:
            print(f"[跳过] 无法解析日期：{os.path.basename(fp)}"); continue
        groups.setdefault(token, []).append(fp)
        centers[token] = center

    if not groups:
        print("[ERROR] 未找到可解析日期的 HDF 文件"); sys.exit(3)

    # 逐日镶嵌
    ok, bad = 0, 0
    for token, flist in sorted(groups.items(), key=lambda kv: centers.get(kv[0], pd.NaT)):
        try:
            # 将每个 HDF 解析为“可读的源”（SDS 或 HDF 自身）
            sd_list = [pick_gpp_subdataset(p) for p in flist]
            out_name = f"GPP_{token}_mosaic.tif"
            out_tif = os.path.join(cfg["OUT_DIR"], out_name)
            if os.path.exists(out_tif):
                print(f"[跳过] 已存在：{out_name}")
                ok += 1
                continue
            mosaic_one_day(
                sd_list=sd_list,
                out_tif=out_tif,
                force_src_crs=cfg["FORCE_SRC_CRS"],
                warp_to_epsg4326=cfg["WARP_TO_EPSG4326"],
                xres=cfg["XRES"],
                yres=cfg["YRES"],
                dst_nodata=cfg["DST_NODATA"],
                resample=cfg["RESAMPLE"],
                compress=cfg["COMPRESS"],
                bigtiff=cfg["BIGTIFF"],
                tiled=cfg["TILED"],
            )
            print(f"[OK] {out_name}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {token} : {e}")
            traceback.print_exc()
            bad += 1

    print(f"\n完成。成功 {ok}，失败 {bad}。输出目录：{cfg['OUT_DIR']}")


if __name__ == "__main__":
    main()
