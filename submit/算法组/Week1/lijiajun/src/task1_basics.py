import cv2 as cv
import numpy as np
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data", "images")
OUT_DIR = os.path.join(BASE, "outputs", "task1")

DEFAULT_IMAGES = ("test5", "test2")


def imread_zh(path, flags=cv.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv.imdecode(data, flags)


def imwrite_zh(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def save_and_check(name, img, show):
    path = os.path.join(OUT_DIR, name)
    ok = imwrite_zh(path, img)
    back = imread_zh(path, cv.IMREAD_UNCHANGED) if ok else None
    same = back is not None and back.shape == img.shape
    size = os.path.getsize(path) if ok else 0
    print(f"  {name:<22} 保存{'成功' if ok else '失败'}  "
          f"重读{'一致' if same else '异常'}  shape={None if back is None else back.shape}  {size / 1024:.0f} KB")
    if show and back is not None:
        cv.imshow(name, back)
    return same


def process(filename, show):
    path = os.path.join(IMG_DIR, f"{filename}.png")
    img = imread_zh(path)
    if img is None:
        print(f"[{filename}] 读不到图片：{path}")
        return False

    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    print(f"\n[{filename}] 高={h} 宽={w} 通道={channels} dtype={img.dtype}")

    small = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    crop = img[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ok = True
    ok &= save_and_check(f"{filename}_resize.png", small, show)
    ok &= save_and_check(f"{filename}_crop.png", crop, show)
    ok &= save_and_check(f"{filename}_gray.png", gray, show)

    print(f"  缩放 {w}x{h} -> {small.shape[1]}x{small.shape[0]}，"
          f"裁剪区域 x={w // 4}~{w // 4 + w // 2} y={h // 4}~{h // 4 + h // 2}，"
          f"灰度 {img.shape[2]} 通道 -> 1 通道")
    print(f"  原图仍为 {img.shape}，未被覆盖")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    names = args or list(DEFAULT_IMAGES)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = [process(n, show) for n in names]
    if show:
        cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"\n{sum(results)}/{len(results)} 张图片处理完毕，输出目录：outputs/task1/")


if __name__ == "__main__":
    main()
