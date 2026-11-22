import cv2 as cv
import numpy as np
import os
import shutil
import sys
import tempfile
import time

import task1_basics as t1
import task3_color_detect as t3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data", "images")
OUT_DIR = os.path.join(BASE, "outputs", "task4")
DEMO_VIDEO = os.path.join(BASE, "data", "videos", "demo.mp4")

DEMO_SIZE = (640, 480)
DEMO_FPS = 15
SMOOTH_ALPHA = 0.35
LOST_LIMIT = 8
SNAP_FRAMES = {15: "snap_normal.png", 45: "snap_wrong_color.png", 75: "snap_dim.png",
               105: "snap_dark.png", 135: "snap_bright.png"}

KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

SEGMENTS = [
    ("test5", 1.00, 2.0),
    ("test6", 1.00, 2.0),
    ("test5", 0.60, 2.0),
    ("test5", 0.45, 2.0),
    ("test5", 1.15, 2.0),
]


def ascii_copy(path):
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    tmp = os.path.join(tempfile.gettempdir(), "task4_src_" + os.path.basename(path))
    shutil.copyfile(path, tmp)
    return tmp


def open_writer(fps, size):
    tmp = os.path.join(tempfile.gettempdir(), "task4_out.mp4")
    writer = cv.VideoWriter(tmp, cv.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        sys.exit("VideoWriter 打不开，输出视频失败")
    return writer, tmp


def detect(frame, preset):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = t3.make_mask(hsv, t3.RANGES[preset])
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    min_area = h * w * t3.AREA_RATIO.get(preset, t3.DEFAULT_RATIO)
    best = None
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area:
            continue
        if best is None or area > best[0]:
            x, y, bw, bh = cv.boundingRect(c)
            best = (area, (x, y, bw, bh), (x + bw // 2, y + bh // 2))
    return best, min_area


def demo_frame(img, shift, gain, w, h, rng):
    base = cv.resize(img, (h, h), interpolation=cv.INTER_AREA).astype(np.float32) * gain
    base = base + rng.normal(0, 3, base.shape)
    base = np.clip(base, 0, 255).astype(np.uint8)
    canvas = np.zeros((h, w, 3), np.uint8)
    x = (w - h) // 2 + shift
    canvas[:, x:x + h] = base
    return canvas


def make_demo():
    os.makedirs(os.path.dirname(DEMO_VIDEO), exist_ok=True)
    w, h = DEMO_SIZE
    writer, tmp = open_writer(DEMO_FPS, DEMO_SIZE)
    rng = np.random.default_rng(0)
    total = 0
    for name, gain, seconds in SEGMENTS:
        img = t1.imread_zh(os.path.join(IMG_DIR, f"{name}.png"))
        if img is None:
            sys.exit(f"读不到图片：{name}.png")
        n = int(DEMO_FPS * seconds)
        for i in range(n):
            shift = int(-60 + 120 * i / max(n - 1, 1))
            writer.write(demo_frame(img, shift, gain, w, h, rng))
            total += 1
        print(f"  {name:<6} 亮度x{gain:.2f}  {n:>3} 帧  {seconds:.1f} 秒")
    writer.release()
    shutil.move(tmp, DEMO_VIDEO)
    print(f"演示视频已生成：{DEMO_VIDEO}  共 {total} 帧，{total / DEMO_FPS:.1f} 秒，"
          f"{os.path.getsize(DEMO_VIDEO) / 1024:.0f} KB")


def run(preset, source, headless):
    if preset not in t3.RANGES:
        sys.exit(f"RANGES 里没有 {preset}，可选：{', '.join(t3.RANGES)}")

    if source.isdigit():
        cap = cv.VideoCapture(int(source))
        fps_src = 30.0
        label = f"camera {source}"
    else:
        if not os.path.exists(source):
            sys.exit(f"视频不存在：{source}")
        cap = cv.VideoCapture(ascii_copy(source))
        fps_src = cap.get(cv.CAP_PROP_FPS) or DEMO_FPS
        label = os.path.basename(source)

    if not cap.isOpened():
        sys.exit(f"打不开视频源：{source}，摄像头被占用或视频编码不支持？")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or DEMO_SIZE[0]
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or DEMO_SIZE[1]

    writer = None
    tmp_out = None
    if headless:
        os.makedirs(OUT_DIR, exist_ok=True)
        writer, tmp_out = open_writer(fps_src, (width, height))
        print(f"[{label}] {width}x{height} @ {fps_src:.0f} FPS，preset={preset}，无窗口模式")

    smooth = None
    raw_prev = None
    smooth_prev = None
    raw_jump = []
    smooth_jump = []
    lost = 0
    lost_run = 0
    frames = hits = 0
    ms_show = 0.0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t0 = time.time()
        best, min_area = detect(frame, preset)

        if best is not None:
            hits += 1
            lost = 0
            area, box, center = best
            if smooth is None:
                smooth = np.array(center, np.float32)
            else:
                smooth = (1 - SMOOTH_ALPHA) * smooth + SMOOTH_ALPHA * np.array(center, np.float32)
            sx, sy = smooth.astype(int)

            if raw_prev is not None:
                d = float(np.hypot(center[0] - raw_prev[0], center[1] - raw_prev[1]))
                ds = float(np.hypot(sx - smooth_prev[0], sy - smooth_prev[1]))
                raw_jump.append(d)
                smooth_jump.append(ds)
            raw_prev = center
            smooth_prev = (sx, sy)

            x, y, bw, bh = box
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, center, 4, (0, 255, 255), -1)
            cv.line(frame, center, (sx, sy), (255, 0, 255), 1)
            cv.circle(frame, (sx, sy), 6, (0, 0, 255), -1)
            cv.putText(frame, f"f{frames} raw({center[0]},{center[1]}) smooth({sx},{sy}) a={area:.0f}",
                       (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            lost += 1
            lost_run += 1
            if lost >= LOST_LIMIT:
                smooth = None
                raw_prev = None
                smooth_prev = None
            cv.putText(frame, f"f{frames} TARGET LOST ({lost_run} frames)", (10, 24),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ms_show = 0.9 * ms_show + 0.1 * (time.time() - t0) * 1000
        info = f"{preset}  proc {ms_show:5.2f} ms  hit {100.0 * hits / frames:4.1f}%  minArea {min_area:.0f}"
        cv.putText(frame, info, (10, height - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                   cv.LINE_AA)

        if headless:
            writer.write(frame)
            if frames in SNAP_FRAMES and os.path.basename(source) == "demo.mp4":
                note = "LOST" if best is None else f"{best[0]:.0f}px"
                path = os.path.join(OUT_DIR, SNAP_FRAMES[frames])
                t1.imwrite_zh(path, frame)
                print(f"  第 {frames:>3} 帧  {SNAP_FRAMES[frames]:<22} {note}")
        else:
            cv.imshow(f"task4 {label}", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t_start
    cap.release()
    if writer is not None:
        writer.release()
        shutil.move(tmp_out, os.path.join(OUT_DIR, "annotated.mp4"))
    cv.destroyAllWindows()

    raw_mean = float(np.mean(raw_jump)) if raw_jump else 0.0
    smooth_mean = float(np.mean(smooth_jump)) if smooth_jump else 0.0
    raw_med = float(np.median(raw_jump)) if raw_jump else 0.0
    smooth_med = float(np.median(smooth_jump)) if smooth_jump else 0.0
    flips = sum(1 for d in raw_jump if d > 50)
    print(f"\n[{label}] 总帧数 {frames}，检出 {hits}，检出率 {100.0 * hits / max(frames, 1):.1f}%")
    print(f"  耗时 {elapsed:.1f} 秒，平均单帧处理 {1000 * elapsed / max(frames, 1):.2f} ms，"
          f"约 {frames / max(elapsed, 1e-6):.0f} FPS 上限")
    print(f"  中心点位移  原始 均值 {raw_mean:.2f} / 中位数 {raw_med:.2f} px每帧")
    print(f"  中心点位移  平滑 均值 {smooth_mean:.2f} / 中位数 {smooth_med:.2f} px每帧")
    print(f"  单帧跳动超过 50px 的帧数 {flips}（误检目标间来回切换）")
    if headless:
        print(f"  标注视频：outputs/task4/annotated.mp4")
    print(f"  失效场景：连续 {LOST_LIMIT} 帧无目标后清空平滑值，避免恢复时从旧位置慢移回来")


def bench_smoothing(name, frames=150, alphas=(0.2, 0.35, 0.5)):
    img = t1.imread_zh(os.path.join(IMG_DIR, f"{name}.png"))
    if img is None:
        sys.exit(f"读不到图片：{name}.png")
    base = cv.resize(img, DEMO_SIZE, interpolation=cv.INTER_AREA).astype(np.float32)
    rng = np.random.default_rng(1)
    centers = []
    for _ in range(frames):
        dx, dy = rng.uniform(-3, 3, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv.warpAffine(base, M, DEMO_SIZE)
        frame = np.clip(shaken + rng.normal(0, 4, shaken.shape), 0, 255).astype(np.uint8)
        best, _ = detect(frame, name)
        if best is not None:
            centers.append(np.array(best[2], np.float32))
    if len(centers) < 2:
        sys.exit(f"{name} 在 {frames} 帧里检出太少，无法统计")

    raw = np.array([np.hypot(*(centers[i] - centers[i - 1])) for i in range(1, len(centers))])
    print(f"[{name}] 模拟手持抖动（每帧随机平移 ±3px）+ 高斯噪声 sigma=4，共 {len(centers)} 帧")
    print(f"  原始中心点位移  均值 {raw.mean():.2f}  中位数 {np.median(raw):.2f}  "
          f"标准差 {raw.std():.2f} px每帧")
    for a in alphas:
        s = centers[0].copy()
        smooth = []
        lag = []
        for c in centers[1:]:
            prev = s.copy()
            s = (1 - a) * s + a * c
            smooth.append(float(np.hypot(*(s - prev))))
            lag.append(float(np.hypot(*(s - c))))
        smooth = np.array(smooth)
        print(f"  alpha={a:<4} 平滑后位移  均值 {smooth.mean():.2f}  中位数 {np.median(smooth):.2f}  "
              f"标准差 {smooth.std():.2f} px每帧  稳态滞后 {np.mean(lag[-40:]):.2f} px")
    print(f"  结论：alpha 越小越稳但滞后越大，{SMOOTH_ALPHA} 是折中值")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--make-demo" in sys.argv:
        make_demo()
        return
    if "--bench" in sys.argv:
        bench_smoothing(args[0] if args else "test5")
        return
    preset = args[0] if args else "test5"
    source = args[1] if len(args) > 1 else DEMO_VIDEO
    if source == DEMO_VIDEO:
        if not os.path.exists(DEMO_VIDEO):
            sys.exit("还没有演示视频，先运行：python src/task4_camera.py --make-demo")
        print(f"没有指定视频源，使用演示视频 data/videos/demo.mp4；用摄像头请传 0")
    run(preset, source, "--headless" in sys.argv)


if __name__ == "__main__":
    main()
