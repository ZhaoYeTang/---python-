# -*- coding: utf-8 -*-
import os
import re
from collections import Counter

# ========= 改这里 =========
FOLDER = r"E:\southWestern-dongmeng\forest\landtrendr\tiles"  # 放 tile tif 的文件夹
PREFIX = "LT_segments_fp_2004_2024_tile_"
SUFFIX = ".tif"
START = 0
END = 2147  # 包含 2147
SAVE_MISSING_TXT = True
MISSING_TXT = os.path.join(FOLDER, "missing_tiles.txt")
# ===========================

pattern = re.compile(rf"^{re.escape(PREFIX)}(\d+){re.escape(SUFFIX)}$", re.IGNORECASE)

files = [f for f in os.listdir(FOLDER) if f.lower().endswith(".tif")]

found_ids = []
unmatched = []
for f in files:
    m = pattern.match(f)
    if m:
        found_ids.append(int(m.group(1)))  # int() 自动去掉前导0
    else:
        # 不是需要的命名
        unmatched.append(f)

cnt = Counter(found_ids)

expected = list(range(START, END + 1))
expected_set = set(expected)
found_set = set(found_ids)

missing = sorted(expected_set - found_set)
extra = sorted(found_set - expected_set)
duplicates = sorted([k for k, v in cnt.items() if v > 1])

print("=== Tile completeness check ===")
print("Folder:", FOLDER)
print(f"Expected range: {START}..{END} (total {len(expected)})")
print(f"Matched files: {len(found_ids)} (unique ids {len(found_set)})")

print("\nMissing count:", len(missing))
if len(missing) <= 200:
    print("Missing ids:", missing)
else:
    print("Missing ids (first 50):", missing[:50])
    print("Missing ids (last 50):", missing[-50:])

print("\nExtra ids (outside range) count:", len(extra))
if extra:
    print("Extra ids:", extra[:100])

print("\nDuplicate ids count:", len(duplicates))
if duplicates:
    print("Duplicate ids:", duplicates[:100])
    for d in duplicates[:10]:
        print(f"  id {d} appears {cnt[d]} times")

print("\nUnmatched tif files (not following naming) count:", len(unmatched))
if len(unmatched) <= 50:
    for f in unmatched:
        print("  ", f)
else:
    print("  (too many to list)")

if SAVE_MISSING_TXT:
    with open(MISSING_TXT, "w", encoding="utf-8") as fw:
        fw.write("\n".join(map(str, missing)))
    print("\nSaved missing list to:", MISSING_TXT)
