import numpy as np
import rasterio
from rasterio.windows import Window

# -----------------------
# 你需要改的输入输出路径
# -----------------------
IN_TIF  = r"E:\southWestern-dongmeng\forest\landtrendr\mosaic3_filled.tif"
OUT_TIF = r"E:\southWestern-dongmeng\forest\landtrendr\LT_metrics3.tif"

# 与你 GEE 代码一致的阈值
MIN_MAG = 0.08
MIN_DUR = 0
EPS = 1e-6

# dominant_sign 定义：
# True: 依据 max_pos_intensity vs max_neg_intensity（推荐）
# False: 依据 n_pos vs n_neg
DOMINANT_BY_MAX_INTENSITY = True


def compute_metrics_from_segments(sy, ey, sv, ev):
    """
    输入:
      sy,ey,sv,ev: shape (6,H,W) float32
    输出:
      9个指标的 float32 数组, shape (9,H,W)
    """
    dur = ey - sy
    mag = ev - sv
    abs_mag = np.abs(mag)

    # intensity = |mag|/dur (dur<=0 的段置 0)
    intensity = np.where(dur > 0, abs_mag / dur, 0.0)

    valid = (dur >= MIN_DUR) & (abs_mag >= MIN_MAG)
    pos = valid & (mag >  MIN_MAG)
    neg = valid & (mag < -MIN_MAG)

    n_all = valid.sum(axis=0).astype(np.float32)
    n_pos = pos.sum(axis=0).astype(np.float32)
    n_neg = neg.sum(axis=0).astype(np.float32)

    # mean_duration_all
    sum_dur = np.where(valid, dur, 0.0).sum(axis=0)
    mean_dur = np.where(n_all > 0, sum_dur / n_all, 0.0).astype(np.float32)

    # max_pos_intensity + year_max_pos（按你 JS：并列最大取 startYear 的最大值）
    pos_int = np.where(pos, intensity, 0.0)
    max_pos = pos_int.max(axis=0).astype(np.float32)

    sel_pos = (np.abs(pos_int - max_pos[None, ...]) < EPS) & pos
    year_pos = np.where(sel_pos, sy, 0.0).max(axis=0).astype(np.float32)

    # max_neg_intensity + year_max_neg
    neg_int = np.where(neg, intensity, 0.0)
    max_neg = neg_int.max(axis=0).astype(np.float32)

    sel_neg = (np.abs(neg_int - max_neg[None, ...]) < EPS) & neg
    year_neg = np.where(sel_neg, sy, 0.0).max(axis=0).astype(np.float32)

    # dominant_sign
    if DOMINANT_BY_MAX_INTENSITY:
        dominant = np.where(max_pos > max_neg, 1,
                   np.where(max_neg > max_pos, -1, 0)).astype(np.float32)
    else:
        dominant = np.where(n_pos > n_neg, 1,
                   np.where(n_neg > n_pos, -1, 0)).astype(np.float32)

    out = np.stack([
        n_pos,                 # 1
        n_neg,                 # 2
        n_all,                 # 3
        dominant,              # 4
        max_pos,               # 5
        max_neg,               # 6
        year_pos,              # 7
        year_neg,              # 8
        mean_dur               # 9
    ], axis=0).astype(np.float32)

    return out


def main():
    with rasterio.open(IN_TIF) as src:
        prof = src.profile.copy()

        if src.count < 24:
            raise RuntimeError(f"Input must have 24 bands (sy/ey/sv/ev x 6). Got {src.count}.")

        # 输出 profile
        out_prof = prof.copy()
        out_prof.update(
            count=9,
            dtype="float32",
            compress="lzw"
        )

        # 为了稳：按 block/windows 处理
        # 如果原图有 tiling/blocking，就按它的 block size；否则用一个默认窗口
        try:
            block_h, block_w = src.block_shapes[0]
        except Exception:
            block_h, block_w = (512, 512)

        width, height = src.width, src.height
        print(f"Input size: {width} x {height}, block: {block_w} x {block_h}")

        with rasterio.open(OUT_TIF, "w", **out_prof) as dst:
            # 写 band 描述（可选）
            dst.set_band_description(1, "n_pos")
            dst.set_band_description(2, "n_neg")
            dst.set_band_description(3, "n_all")
            dst.set_band_description(4, "dominant_sign")
            dst.set_band_description(5, "max_pos_intensity")
            dst.set_band_description(6, "max_neg_intensity")
            dst.set_band_description(7, "year_max_pos")
            dst.set_band_description(8, "year_max_neg")
            dst.set_band_description(9, "mean_duration_all")

            # 遍历窗口
            for row_off in range(0, height, block_h):
                win_h = min(block_h, height - row_off)
                for col_off in range(0, width, block_w):
                    win_w = min(block_w, width - col_off)
                    window = Window(col_off, row_off, win_w, win_h)

                    # 读 24 bands: [0:6]=sy, [6:12]=ey, [12:18]=sv, [18:24]=ev
                    data = src.read(window=window).astype(np.float32)

                    sy = data[0:6]
                    ey = data[6:12]
                    sv = data[12:18]
                    ev = data[18:24]

                    out = compute_metrics_from_segments(sy, ey, sv, ev)

                    dst.write(out, window=window)

                print(f"Processed rows {row_off}..{row_off+win_h-1}")

    print("Done:", OUT_TIF)


if __name__ == "__main__":
    main()