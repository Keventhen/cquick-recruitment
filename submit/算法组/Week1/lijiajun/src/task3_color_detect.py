import cv2 as cv
import numpy as np
import os
import sys

import task1_basics as t1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data", "images")
OUT_DIR = os.path.join(BASE, "outputs", "task3")

RANGES = {
    "test2": [((20, 100, 100), (35, 255, 255))],
    "test4": [((5, 150, 100), (20, 255, 255))],
    "test5": [((4, 120, 140), (20, 255, 255))],
    "test6": [((88, 40, 40), (118, 255, 255))],
    "test7": [((35, 60, 40), (85, 255, 255))],
}

AREA_RATIO = {
    "test6": 0.010,
}

DEFAULT_RATIO = 0.001

KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


def make_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        part = cv.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        mask = cv.bitwise_or(mask, part)
    return mask


def run(filename, show):
    path = os.path.join(IMG_DIR, f"{filename}.png")
    img = t1.imread_zh(path)
    if img is None:
        print(f"[{filename}] 读不到图片：{path}")
        return 0

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    raw = make_mask(hsv, RANGES[filename])
    mask = cv.morphologyEx(raw, cv.MORPH_OPEN, KERNEL)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape[:2]
    min_area = h * w * AREA_RATIO.get(filename, DEFAULT_RATIO)
    targets = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv.boundingRect(c)
        targets.append({
            "area": area,
            "box": (x, y, bw, bh),
            "center": (x + bw // 2, y + bh // 2),
        })
    targets.sort(key=lambda t: t["area"], reverse=True)

    print(f"\n[{filename}] {w}x{h}  inRange={cv.countNonZero(raw)}  "
          f"开运算后={cv.countNonZero(mask)}  轮廓={len(contours)}  "
          f"面积阈值={min_area:.0f}  目标={len(targets)}")

    result = img.copy()
    for i, t in enumerate(targets, 1):
        x, y, bw, bh = t["box"]
        cx, cy = t["center"]
        cv.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv.circle(result, (cx, cy), 6, (0, 0, 255), -1)
        cv.putText(result, f"#{i}", (x, max(y - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"    #{i} 中心=({cx}, {cy})  面积={t['area']:.1f}  矩形=({x}, {y}, {bw}, {bh})")

    os.makedirs(OUT_DIR, exist_ok=True)
    mask_path = os.path.join(OUT_DIR, f"{filename}_mask.png")
    result_path = os.path.join(OUT_DIR, f"{filename}_result.png")
    ok_mask = t1.imwrite_zh(mask_path, mask)
    ok_result = t1.imwrite_zh(result_path, result)
    print(f"    {filename}_mask.png {'成功' if ok_mask else '失败'}，"
          f"{filename}_result.png {'成功' if ok_result else '失败'}  "
          f"{os.path.getsize(result_path) / 1024:.0f} KB")

    if show:
        cv.imshow(f"{filename} mask", mask)
        cv.imshow(f"{filename} result", result)
    return len(targets)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    names = args or list(RANGES)

    for n in names:
        if n not in RANGES:
            sys.exit(f"RANGES 里没有 {n}，可选：{', '.join(RANGES)}")

    total = [run(n, show) for n in names]
    if show:
        cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"\n{len(names)} 张图处理完毕，输出目录：outputs/task3/，各图目标数 {total}")


if __name__ == "__main__":
    main()
