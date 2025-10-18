# -*- coding: utf-8 -*-
"""
将 NetCDF(.nc) 中的变量转换为 GeoTIFF(.tif)
- 在 Python 环境中直接运行（Jupyter/Spyder/PyCharm 等）
- 使用 tkinter 弹窗选择 nc 文件与输出目录，并输入变量名
- 自动判断是否包含时间维：有 -> 按时间逐一输出，多张；无 -> 输出单张
"""

import os
import warnings
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import rioxarray
import pandas as pd

# ===== 可选：简单 GUI 选择 =====
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False


def _find_lat_lon_dims(da: xr.DataArray) -> Tuple[str, str]:
    """尽量稳健地在 DataArray 的维度/坐标里找到纬度和经度维名。"""
    cand_lat = ["lat", "latitude", "y"]
    cand_lon = ["lon", "longitude", "x"]

    # 先在 dims 里找
    lat_name = next((d for d in da.dims if d.lower() in cand_lat), None)
    lon_name = next((d for d in da.dims if d.lower() in cand_lon), None)

    # 再在 coords 里找
    if lat_name is None:
        lat_name = next((c for c in da.coords if c.lower() in cand_lat), None)
    if lon_name is None:
        lon_name = next((c for c in da.coords if c.lower() in cand_lon), None)

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"未找到经纬度维度/坐标。检测到的维度: {list(da.dims)}, 坐标: {list(da.coords)}\n"
            f"请确认数据含有 lat/lon 或 latitude/longitude 或 x/y 等。"
        )
    return lat_name, lon_name


def _find_time_dim(da: xr.DataArray) -> Optional[str]:
    """返回时间维名称（如存在）。"""
    for d in da.dims:
        dl = d.lower()
        if ("time" in dl) or (dl in ["t", "month", "date"]):
            return d
    return None


def _format_time_label(val: np.datetime64) -> str:
    """时间标签：YYYYMMDD 或 YYYYMM（如果是月尺度也能兼容）。"""
    ts = pd.to_datetime(val)
    # 若是月尺度、只有月末日期，可输出 YYYYMM
    if ts.day in (28, 29, 30, 31):  # 不严格判断，直接给 YYYYMM 更通用
        return ts.strftime("%Y%m")
    return ts.strftime("%Y%m%d")


def _ensure_crs_and_dims(da: xr.DataArray, lon_name: str, lat_name: str):
    """设置空间维并写入 WGS84 坐标系。"""
    # 如果是坐标在 coords 里但不在 dims 里，确保 DataArray 是二维网格
    if lon_name not in da.dims or lat_name not in da.dims:
        # 常见于经纬度在 coords，数据 dims 是（y, x）且 coords 关联
        pass
    # 设置空间维
    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    # 写 CRS
    if not da.rio.crs:
        da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da


def convert_nc_to_tiff(
    nc_path: str,
    var_name: str,
    output_dir: Optional[str] = None,
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
) -> List[str]:
    """
    将指定 nc 文件中的变量转换为 GeoTIFF。
    返回生成的文件路径列表。

    参数
    ----
    nc_path : str
        .nc 文件路径
    var_name : str
        变量名
    output_dir : str, optional
        输出文件夹；默认与 nc 同级
    nodata : float, optional
        指定输出 GeoTIFF 的 NoData 值（默认不强制写）
    dtype : str, optional
        强制输出数据类型（如 'float32', 'int16'）；默认保持原 dtype
    """
    if output_dir is None or output_dir.strip() == "":
        output_dir = os.path.dirname(nc_path)
    os.makedirs(output_dir, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        ds = xr.open_dataset(nc_path)

    if var_name not in ds.variables:
        raise ValueError(
            f"变量 '{var_name}' 不存在。可选变量：\n{list(ds.variables)}"
        )

    da = ds[var_name]

    # 尝试解码缩放与偏移（如果有）
    # xarray 会自动解码 CF 标准的 scale_factor/add_offset；若未生效，这里显式触发
    if "scale_factor" in da.encoding or "add_offset" in da.encoding:
        da = xr.decode_cf(xr.Dataset({var_name: da}))[var_name]

    # 找经纬度维名
    lat_name, lon_name = _find_lat_lon_dims(da)

    # 找时间维（可无）
    time_dim = _find_time_dim(da)

    # 准备输出
    out_paths = []

    if time_dim is None:
        # 无时间维：输出一张
        da2 = _ensure_crs_and_dims(da.squeeze(drop=True), lon_name, lat_name)
        if nodata is not None:
            da2 = da2.rio.write_nodata(nodata, inplace=False)
        if dtype:
            da2 = da2.astype(dtype)

        out_path = os.path.join(output_dir, f"{var_name}.tif")
        da2.rio.to_raster(out_path)
        out_paths.append(out_path)

    else:
        # 有时间维：逐一输出
        for t in da[time_dim].values:
            slice_da = da.sel({time_dim: t}).squeeze(drop=True)
            slice_da = _ensure_crs_and_dims(slice_da, lon_name, lat_name)
            if nodata is not None:
                slice_da = slice_da.rio.write_nodata(nodata, inplace=False)
            if dtype:
                slice_da = slice_da.astype(dtype)

            t_str = _format_time_label(t)
            out_path = os.path.join(output_dir, f"{var_name}_{t_str}.tif")
            slice_da.rio.to_raster(out_path)
            out_paths.append(out_path)

    ds.close()
    return out_paths


# ===== 下面是“无需终端”的可视化交互（可选）=====
def run_with_gui():
    if not _HAS_TK:
        raise RuntimeError("本机未安装 tkinter，无法使用图形界面。可直接在代码里调用 convert_nc_to_tiff(...)。")

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("nc -> GeoTIFF", "请选择一个 .nc 文件")
    nc_path = filedialog.askopenfilename(
        title="选择 .nc 文件",
        filetypes=[("NetCDF", "*.nc"), ("所有文件", "*.*")]
    )
    if not nc_path:
        messagebox.showwarning("取消", "未选择文件。")
        return

    # 打开数据并列出变量，提示输入
    ds = xr.open_dataset(nc_path)
    var_list = list(ds.variables)
    ds.close()

    var_tip = "可选变量：\n- " + "\n- ".join(var_list)
    var_name = simpledialog.askstring("变量名", f"请输入需要转换的变量名：\n\n{var_tip}")
    if not var_name:
        messagebox.showwarning("取消", "未输入变量名。")
        return

    messagebox.showinfo("输出目录", "请选择输出文件夹")
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        # 若未选则默认与 nc 同目录
        output_dir = os.path.dirname(nc_path)

    # 可选：设置 NoData 和 dtype
    nodata_str = simpledialog.askstring("NoData（可留空）", "为 GeoTIFF 设置 NoData 值（例如 -9999），留空表示不设置：")
    nodata = float(nodata_str) if nodata_str and nodata_str.strip() != "" else None

    dtype = simpledialog.askstring("输出数据类型（可留空）", "如需强制输出 dtype，请输入（例如 float32、int16）。留空表示保持原始类型：")
    dtype = dtype.strip() if dtype else None

    try:
        paths = convert_nc_to_tiff(nc_path, var_name, output_dir, nodata=nodata, dtype=dtype)
        msg = "转换完成！\n\n输出文件：\n" + "\n".join(paths)
        messagebox.showinfo("成功", msg)
    except Exception as e:
        messagebox.showerror("错误", str(e))


# ===== 使用示例 =====
if __name__ == "__main__":
    """
    方式 A：直接在代码里调用（完全不需要终端交互）
        convert_nc_to_tiff(
            nc_path=r"E:\path\to\your.nc",
            var_name="Et",
            output_dir=r"E:\path\to\out",
            nodata=-9999,
            dtype="float32"
        )

    方式 B：使用 GUI 点选（如果你装了 tkinter）
        run_with_gui()
    """
    # 👉 默认走 GUI；如不需要，注释掉下一行，改用“方式 A”示例。
    if _HAS_TK:
        run_with_gui()
    else:
        # 没有 tkinter 的环境，给出一个最简示例（请修改路径与变量）
        # convert_nc_to_tiff(r"/path/to/your.nc", "Et", r"/path/to/out", nodata=-9999, dtype="float32")
        pass
