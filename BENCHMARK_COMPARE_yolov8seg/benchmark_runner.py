#!/usr/bin/env python3
"""
Runtime Benchmark - preprocess_image (Manual letterbox vs cv2.dnn.blobFromImage)
"""

import time
import gc
import sys
import subprocess
import json
from datetime import datetime
from dataclasses import dataclass

def install_packages():
    for pkg in ['psutil', 'numpy', 'opencv-python']:
        try:
            __import__(pkg.replace('-', '_').split('.')[0])
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
        return self.max_mem, self.max_cpu

    def avg_cpu(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0


# =============================================================================
# 版本 A: 手動 letterbox + transforms
# =============================================================================
class PreprocessA:
    def __init__(self, model_height=640, model_width=640):
        self.model_height = model_height
        self.model_width = model_width

    def preprocess(self, img_bgr: np.ndarray):
        """
        Pre-processes the input image.

        Args:
            img_bgr (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img_bgr.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img_bgr = np.ascontiguousarray(np.einsum('HWC->CHW', img_bgr)[::-1], dtype=np.single) / 255.0
        img_process = img_bgr[None] if len(img_bgr.shape) == 3 else img_bgr
        return img_process, ratio, (pad_w, pad_h)


# =============================================================================
# 版本 B: cv2.dnn.blobFromImage
# =============================================================================
class PreprocessB:
    def __init__(self, model_height=640, model_width=640):
        self.model_height = model_height
        self.model_width = model_width

    def preprocess(self, img_bgr: np.ndarray):
        """Fast pre‑processing using OpenCV's ``blobFromImage``."""
        h, w = img_bgr.shape[:2]
        r = min(self.model_height / h, self.model_width / w)
        blob = cv2.dnn.blobFromImage(
            img_bgr,
            scalefactor=1 / 255.0,
            size=(self.model_width, self.model_height),
            mean=(114, 114, 114),
            swapRB=True,
            crop=False,
        )
        img_process = blob.astype(np.single)
        new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
        pad_w = (self.model_width - new_unpad_w) / 2.0
        pad_h = (self.model_height - new_unpad_h) / 2.0
        ratio = (r, r)
        return img_process, ratio, (pad_w, pad_h)


# =============================================================================
# Benchmark
# =============================================================================
IMAGE_SIZES = [(480, 640), (720, 1280), (1080, 1920), (1440, 2560)]
CALLS = [1, 10, 100, 1000]  # 高頻率測試


def benchmark(preprocessor, image, n_calls, warmup=2):
    gc.collect()
    time.sleep(0.1)

    # Warmup
    for _ in range(warmup):
        try:
            _ = preprocessor.preprocess(image)
        except:
            pass
    gc.collect()

    gc.collect()
    time.sleep(0.1)

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()

    success = True
    try:
        for _ in range(n_calls):
            _ = preprocessor.preprocess(image)
    except Exception as e:
        success = False
        print(f"Error: {e}")

    t1 = time.perf_counter()
    mem, cpu = mon.stop()
    avg_cpu = mon.avg_cpu()

    return (t1 - t0) * 1000 / n_calls, mem, cpu, avg_cpu, success


def main():
    print("=" * 80)
    print("   preprocess Benchmark (Manual letterbox vs cv2.dnn.blobFromImage)")
    print("   A: Manual letterbox + transforms")
    print("   B: cv2.dnn.blobFromImage")
    print("=" * 80)

    all_results = []
    preprocess_a = PreprocessA()
    preprocess_b = PreprocessB()

    for size in IMAGE_SIZES:
        for n_calls in CALLS:
            print(f"\n{'─'*80}")
            print(f"  Size: {size[1]}x{size[0]}, Calls: {n_calls}")
            print(f"{'─'*80}")

            # 產生測試影像
            image_bgr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

            ra = benchmark(preprocess_a, image_bgr, n_calls)
            rb = benchmark(preprocess_b, image_bgr, n_calls)

            winner = "A" if ra[0] < rb[0] else "B"
            diff_pct = abs(ra[0] - rb[0]) / max(ra[0], rb[0]) * 100

            print(f"    A: {ra[0]:.4f} ms  |  Mem: {ra[1]:.1f} MB  |  CPU: {ra[2]:.1f}%  |  AvgCPU: {ra[3]:.1f}%  {'✓' if ra[4] else '✗'}")
            print(f"    B: {rb[0]:.4f} ms  |  Mem: {rb[1]:.1f} MB  |  CPU: {rb[2]:.1f}%  |  AvgCPU: {rb[3]:.1f}%  {'✓' if rb[4] else '✗'}")
            print(f"    Winner: {winner}  (diff: {diff_pct:.1f}%)")

            all_results.append({
                "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "a_ms": ra[0],
                "b_ms": rb[0],
                "a_mem": ra[1],
                "b_mem": rb[1],
                "a_cpu": ra[2],
                "b_cpu": rb[2],
                "a_avg_cpu": ra[3],
                "b_avg_cpu": rb[3],
                "winner": winner,
                "diff_pct": diff_pct
            })

            gc.collect()
            time.sleep(0.1)

    # Summary
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)

    a_wins_total = sum(1 for r in all_results if r["winner"] == "A")
    b_wins_total = sum(1 for r in all_results if r["winner"] == "B")
    avg_a_total = sum(r["a_ms"] for r in all_results) / len(all_results)
    avg_b_total = sum(r["b_ms"] for r in all_results) / len(all_results)
    avg_a_mem = sum(r["a_mem"] for r in all_results) / len(all_results)
    avg_b_mem = sum(r["b_mem"] for r in all_results) / len(all_results)
    avg_a_cpu = sum(r["a_cpu"] for r in all_results) / len(all_results)
    avg_b_cpu = sum(r["b_cpu"] for r in all_results) / len(all_results)

    print(f"\n  Overall win count:")
    print(f"    A wins: {a_wins_total}/{len(all_results)}")
    print(f"    B wins: {b_wins_total}/{len(all_results)}")

    print(f"\n  Avg Runtime:")
    print(f"    A: {avg_a_total:.4f} ms")
    print(f"    B: {avg_b_total:.4f} ms")
    print(f"    Winner: {'A' if avg_a_total < avg_b_total else 'B'}  (diff: {abs(avg_a_total-avg_b_total)/max(avg_a_total,avg_b_total)*100:.1f}%)")

    print(f"\n  Avg Memory:")
    print(f"    A: {avg_a_mem:.1f} MB")
    print(f"    B: {avg_b_mem:.1f} MB")
    print(f"    Winner: {'A' if avg_a_mem < avg_b_mem else 'B'}")

    print(f"\n  Avg CPU:")
    print(f"    A: {avg_a_cpu:.1f}%")
    print(f"    B: {avg_b_cpu:.1f}%")
    print(f"    Winner: {'A' if avg_a_cpu < avg_b_cpu else 'B'}")

    # By call frequency
    print(f"\n  By Call Frequency:")
    for calls in [1, 10, 100, 1000]:
        subset = [r for r in all_results if r["calls"] == calls]
        a_wins = sum(1 for r in subset if r["winner"] == "A")
        b_wins = sum(1 for r in subset if r["winner"] == "B")
        avg_a = sum(r["a_ms"] for r in subset) / len(subset)
        avg_b = sum(r["b_ms"] for r in subset) / len(subset)
        avg_a_m = sum(r["a_mem"] for r in subset) / len(subset)
        avg_b_m = sum(r["b_mem"] for r in subset) / len(subset)
        print(f"    x{calls:4d}: A wins {a_wins}/{len(subset)} (avg {avg_a:.4f}ms, {avg_a_m:.1f}MB) | B wins {b_wins}/{len(subset)} (avg {avg_b:.4f}ms, {avg_b_m:.1f}MB)")

    # Save
    with open("preprocess_benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "a_wins": a_wins_total,
                "b_wins": b_wins_total,
                "avg_a_ms": avg_a_total,
                "avg_b_ms": avg_b_total,
                "avg_a_mem": avg_a_mem,
                "avg_b_mem": avg_b_mem,
                "avg_a_cpu": avg_a_cpu,
                "avg_b_cpu": avg_b_cpu
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: preprocess_benchmark_report.json")
    print("=" * 80)


if __name__ == "__main__":
    main()