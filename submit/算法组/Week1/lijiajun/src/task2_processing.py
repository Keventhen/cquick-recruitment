import cv2 as cv
import numpy as np
import os
import sys

import task1_basics as t1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data", "images")
OUT_DIR = os.path.join(BASE, "outputs", "task2")

DEFAULT_IMAGE = "test2"
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


def ratio(binary, total):
    return 100.0 * cv.countNonZero(binary) / total


def delta(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def save(name, img):
    path = os.path.join(OUT_DIR, name)
    ok = t1.imwrite_zh(path, img)
    size = os.path.getsize(path) if ok else 0
    print(f"  {name:<26} {'成功' if ok else '失败'}  {img.shape}  {size / 1024:.0f} KB")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    name = args[0] if args else DEFAULT_IMAGE

    path = os.path.join(IMG_DIR, f"{name}.png")
    img = t1.imread_zh(path)
    if img is None:
        sys.exit(f"读不到图片：{path}")

    os.makedirs(OUT_DIR, exist_ok=True)
    h, w = img.shape[:2]
    total = h * w
    print(f"[{name}] 高={h} 宽={w} 通道={img.shape[2]}  总像素={total}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gauss3 = cv.GaussianBlur(gray, (3, 3), 0)
    gauss7 = cv.GaussianBlur(gray, (7, 7), 0)

    otsu_thresh, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, fixed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, fixed_lo = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
    _, fixed_hi = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)

    canny_hi = cv.Canny(gray, 100, 200)
    canny_lo = cv.Canny(gray, 50, 150)

    opened = cv.morphologyEx(otsu, cv.MORPH_OPEN, KERNEL)
    closed = cv.morphologyEx(otsu, cv.MORPH_CLOSE, KERNEL)

    print("\n保存中间结果：")
    save(f"{name}_gray.png", gray)
    save(f"{name}_gauss3.png", gauss3)
    save(f"{name}_gauss7.png", gauss7)
    save(f"{name}_otsu.png", otsu)
    save(f"{name}_thresh80.png", fixed_lo)
    save(f"{name}_thresh127.png", fixed)
    save(f"{name}_thresh180.png", fixed_hi)
    save(f"{name}_canny100_200.png", canny_hi)
    save(f"{name}_canny50_150.png", canny_lo)
    save(f"{name}_open.png", opened)
    save(f"{name}_close.png", closed)

    print("\n参数对比数据：")
    print(f"  灰度均值            {gray.mean():.1f}")
    print(f"  Otsu 阈值           {otsu_thresh:.0f}")
    print(f"  高斯 3x3 平均变化   {delta(gray, gauss3):.2f}")
    print(f"  高斯 7x7 平均变化   {delta(gray, gauss7):.2f}")
    print(f"  Otsu 白像素占比     {ratio(otsu, total):.2f}%")
    print(f"  固定 80 白像素占比  {ratio(fixed_lo, total):.2f}%")
    print(f"  固定 127 白像素占比 {ratio(fixed, total):.2f}%")
    print(f"  固定 180 白像素占比 {ratio(fixed_hi, total):.2f}%")
    print(f"  Canny 100/200 占比  {ratio(canny_hi, total):.2f}%")
    print(f"  Canny 50/150 占比   {ratio(canny_lo, total):.2f}%")
    print(f"  开运算 5x5 白像素   {cv.countNonZero(otsu)} -> {cv.countNonZero(opened)}")
    print(f"  闭运算 5x5 白像素   {cv.countNonZero(otsu)} -> {cv.countNonZero(closed)}")

    if show:
        for tag in ("gray", "gauss3", "gauss7", "otsu", "thresh80", "thresh127", "thresh180",
                    "canny100_200", "canny50_150", "open", "close"):
            im = t1.imread_zh(os.path.join(OUT_DIR, f"{name}_{tag}.png"), cv.IMREAD_UNCHANGED)
            if im is not None:
                cv.imshow(tag, im)
        cv.waitKey(0)
    cv.destroyAllWindows()
    print("\n输出目录：outputs/task2/")


if __name__ == "__main__":
    main()
