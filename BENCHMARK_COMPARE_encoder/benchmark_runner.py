#!/usr/bin/env python3
"""
Runtime Benchmark - extract_image_patch v3 與 原始版本比對
"""

import time
import gc
import sys
import subprocess
import json
from datetime import datetime
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import traceback

def install_packages():
    for pkg in ['psutil', 'numpy', 'opencv-python']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import numpy as np
import cv2


class ResourceMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.running = False
        self.thread = None
        self.max_mem = 0.0
        self.max_cpu = 0.0
        self.cpu_samples = []
        self.process = None

    def start(self):
        self.running = True
        self.max_mem = 0.0
        self.max_cpu = 0.0
        self.cpu_samples = []
        self.process = psutil.Process()
        import threading
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.running:
                try:
                    self.max_mem = max(self.max_mem, self.process.memory_info().rss / (1024*1024))
                    cpu = psutil.cpu_percent(interval=None)
                    self.max_cpu = max(self.max_cpu, cpu)
                    self.cpu_samples.append(cpu)
                except: pass
                time.sleep(self.interval)
        except: pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.max_mem, self.max_cpu, self.cpu_samples


# =============================================================================
# 版本 A: 原始版本 (類別方法)
# =============================================================================
class PatchExtractor:
    def extract_image_patch(self, img_bgr: np.ndarray, box: list[int], patch_shape: tuple[int, int] = None):
        """原始版本"""
        bbox = np.array(box)
        if patch_shape is not None:
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int32)

        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(img_bgr.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        img_bgr = cv2.resize(img_bgr[sy:ey, sx:ex], tuple(patch_shape[::-1]))
        return img_bgr


# =============================================================================
# 版本 B: Speed King v3 (Standalone)
# =============================================================================
def extract_image_patch_v3(img_bgr: np.ndarray, box: list[int], patch_shape: tuple[int, int] = None) -> np.ndarray | None:
    """Version 3: 極速版本 - 中心點定位 + 自動插值"""
    bbox = np.asarray(box, dtype=np.float32)

    if patch_shape is not None:
        target_aspect = patch_shape[1] / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) * 0.5
        bbox[2] = new_width

    x, y, bw, bh = bbox
    cx, cy = x + bw * 0.5, y + bh * 0.5

    h, w = img_bgr.shape[:2]
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    if patch_shape is None:
        out_size = (max(1, int(bw)), max(1, int(bh)))
    else:
        out_size = (patch_shape[1], patch_shape[0])

    half_w = max(1, out_size[0] // 2)
    half_h = max(1, out_size[1] // 2)

    x1 = max(0, int(cx) - half_w)
    y1 = max(0, int(cy) - half_h)
    x2 = min(w, int(cx) + half_w)
    y2 = min(h, int(cy) + half_h)

    if x1 >= x2 or y1 >= y2:
        return None

    patch = img_bgr[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    interp = cv2.INTER_AREA if (patch.shape[1] > out_size[0] or patch.shape[0] > out_size[1]) else cv2.INTER_LINEAR

    return cv2.resize(patch, out_size, interpolation=interp)


@dataclass
class BenchResult:
    name: str
    runtime_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    shape: str = ""
    error: str = ""


def benchmark_func(func: Callable, image: np.ndarray, bbox: List[int],
                   patch_shape, n_calls: int, warmup: int = 2) -> BenchResult:
    gc.collect()
    time.sleep(0.1)

    # Warmup
    for _ in range(warmup):
        try:
            _ = func(image, bbox, patch_shape)
        except:
            pass
        gc.collect()

    gc.collect()
    time.sleep(0.1)

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()

    ret_val = None
    success = True
    error = ""

    try:
        for _ in range(n_calls):
            ret_val = func(image, bbox, patch_shape)
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    t1 = time.perf_counter()
    mem, cpu, _ = mon.stop()

    shape = ""
    if ret_val is not None and isinstance(ret_val, np.ndarray):
        shape = str(ret_val.shape)

    return BenchResult(name=func.__name__, runtime_ms=(t1 - t0) * 1000 / n_calls,
                      memory_mb=mem, cpu_percent=cpu, success=success, shape=shape, error=error)


PATCH_SHAPE = (128, 64)

# 測試案例
TESTS = [
    ("720p x1",    (720, 1280),    1),
    ("720p x100",  (720, 1280),  100),
    ("720p x1000", (720, 1280), 1000),
    ("1080p x1",   (1080, 1920),   1),
    ("1080p x100", (1080, 1920), 100),
    ("1080p x1000",(1080, 1920),1000),
    ("2K x1",      (1440, 2560),   1),
    ("2K x100",    (1440, 2560), 100),
    ("2K x1000",   (1440, 2560),1000),
    ("4K x1",      (2160, 3840),   1),
    ("4K x100",    (2160, 3840), 100),
    ("4K x1000",   (2160, 3840),1000),
]

BOX = [400, 200, 800, 1200]  # 測試用邊界框
extractor = PatchExtractor()


def main():
    print("=" * 70)
    print("   extract_image_patch vs v3 Benchmark")
    print("=" * 70)

    results = []

    for name, size, calls in TESTS:
        print(f"\n{'─'*70}")
        print(f"  {name}")
        print(f"─"*70)

        image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        # 版本 A (類別方法)
        ra = benchmark_func(extractor.extract_image_patch, image, BOX, PATCH_SHAPE, calls)
        ra.name = "A_原始版本"

        # 版本 B (Standalone)
        rb = benchmark_func(extract_image_patch_v3, image, BOX, PATCH_SHAPE, calls)
        rb.name = "B_v3"

        # 版本 B
        rb = benchmark_func(extract_image_patch_v3, image, BOX, PATCH_SHAPE, calls)

        print(f"    A (original):  {ra.runtime_ms:.4f} ms  |  {ra.memory_mb:.1f} MB  {'✓' if ra.success else '✗'}")
        print(f"    B (v3):        {rb.runtime_ms:.4f} ms  |  {rb.memory_mb:.1f} MB  {'✓' if rb.success else '✗'}")

        winner = "B" if rb.runtime_ms < ra.runtime_ms else "A"
        diff_pct = abs(ra.runtime_ms - rb.runtime_ms) / ra.runtime_ms * 100
        print(f"    Winner: {winner}  (diff: {diff_pct:.1f}%)")

        results.append({
            "test": name,
            "runtime_a_ms": ra.runtime_ms,
            "runtime_b_ms": rb.runtime_ms,
            "winner": winner,
            "diff_pct": diff_pct
        })

        gc.collect()
        time.sleep(0.1)

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)

    a_wins = sum(1 for r in results if r["winner"] == "A")
    b_wins = sum(1 for r in results if r["winner"] == "B")

    avg_a = sum(r["runtime_a_ms"] for r in results) / len(results)
    avg_b = sum(r["runtime_b_ms"] for r in results) / len(results)

    print(f"\n  A wins: {a_wins}/12  (avg: {avg_a:.4f} ms)")
    print(f"  B wins: {b_wins}/12  (avg: {avg_b:.4f} ms)")
    print(f"\n  Overall: {'B' if avg_b < avg_a else 'A'} is faster by {abs(avg_a - avg_b) / max(avg_a, avg_b) * 100:.1f}%")

    # Save
    with open("v3_comparison_report.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()