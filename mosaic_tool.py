# -*- coding: utf-8 -*-
"""
将一个文件夹中的所有栅格镶嵌为一个输出栅格（把 0 当成 NoData 忽略，从而填补条带）
依赖：
    pip install rasterio numpy
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge

# =========================
# 1. 输入输出路径
# =========================
input_folder = r"C:\Users\Tangyi\Downloads\lt0-1154"  # 改成你的栅格文件夹
output_raster = r"E:\southWestern-dongmeng\forest\landtrendr\mosaic3_filled.tif"

# 把这个值当成“无数据”来忽略（你的缺失条带显示为 0）
NODATA_VALUE = 0

# =========================
# 2. 搜索所有栅格
# =========================
raster_files = []
for ext in ["*.tif", "*.tiff", "*.img"]:
    raster_files.extend(glob.glob(os.path.join(input_folder, ext)))

if len(raster_files) == 0:
    raise FileNotFoundError("输入文件夹中没有找到栅格文件，请检查路径和文件格式。")

# 排序保证拼接顺序稳定
raster_files = sorted(raster_files)

print(f"找到 {len(raster_files)} 个栅格文件。")

# =========================
# 3. 打开所有栅格
# =========================
src_files = [rasterio.open(fp) for fp in raster_files]

# （可选）检查 band 数量一致
counts = {src.count for src in src_files}
if len(counts) != 1:
    for fp, src in zip(raster_files, src_files):
        print(fp, "bands=", src.count)
    raise RuntimeError("输入栅格 band 数量不一致，无法直接 mosaic。")

# =========================
# 4. 执行镶嵌：关键是 nodata=0
# =========================
# nodata 参数会让 merge 把 nodata 当空值，不参与覆盖
mosaic, out_trans = merge(
    src_files,
    nodata=NODATA_VALUE,
    method="first"   # 默认就是 first：按顺序用“第一个非 nodata”的值
)

# =========================
# 5. 继承元数据并更新
# =========================
out_meta = src_files[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "count": mosaic.shape[0],
    "compress": "lzw",
    "nodata": NODATA_VALUE
})

os.makedirs(os.path.dirname(output_raster), exist_ok=True)

# =========================
# 6. 保存结果
# =========================
with rasterio.open(output_raster, "w", **out_meta) as dest:
    dest.write(mosaic)

# =========================
# 7. 关闭
# =========================
for src in src_files:
    src.close()

print(f"\n镶嵌完成（忽略 0 填补空洞），输出文件为：{output_raster}")