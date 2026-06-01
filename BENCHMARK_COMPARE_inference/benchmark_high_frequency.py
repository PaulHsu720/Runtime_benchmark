#!/usr/bin/env python3
"""
Benchmark: High Frequency Load Test
測試高頻率呼叫下的效能差異
"""

import time
import gc
import sys
import subprocess
import json
from datetime import datetime

def install_packages():
    for pkg in ['psutil', 'numpy', 'opencv-python', 'pillow']:
        try:
            __import__(pkg.replace('-', '_').split('.')[0])
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T


def get_transform():
    return T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# Version A: PIL-based
# =============================================================================
def load_image_a(path):
    transform = get_transform()
    img = Image.open(path).convert("RGB")
    np_img = np.asarray(img)
    transformed, _ = transform(img, None)
    return np_img, transformed

def image_preprocess_a(img, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transformed, _ = transform(img, None)
    return transformed

def preprocess_image_a(img_bgr):
    transform = get_transform()
    pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    transformed, _ = transform(pillow, None)
    return transformed


# =============================================================================
# Version B: OpenCV + dtype check
# =============================================================================
def load_image_b(path):
    transform = get_transform()
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    pillow = Image.fromarray(rgb)
    transformed, _ = transform(pillow, None)
    return rgb, transformed

def image_preprocess_b(img, transform):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    pillow = Image.fromarray(rgb)
    transformed, _ = transform(pillow, None)
    return transformed

def preprocess_image_b(img_bgr):
    transform = get_transform()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    pillow = Image.fromarray(rgb)
    transformed, _ = transform(pillow, None)
    return transformed


# =============================================================================
# Monitor
# =============================================================================
class Monitor:
    def __init__(self):
        self.running = False
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
                    mem = self.process.memory_info().rss / 1024**2
                    cpu = psutil.cpu_percent(interval=None)
                    self.max_mem = max(self.max_mem, mem)
                    self.max_cpu = max(self.max_cpu, cpu)
                    self.cpu_samples.append(cpu)
                except: pass
                time.sleep(0.005)
        except: pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        return self.max_mem, self.max_cpu, avg_cpu


# =============================================================================
# Benchmark
# =============================================================================
IMAGE_SIZES = [(480, 640), (720, 1280), (1080, 1920), (1440, 2560)]
# 高頻率測試
HIGH_CALLS = [1000, 5000, 10000, 50000]


def benchmark(func, *args, n_calls=10000, warmup=5):
    gc.collect()
    time.sleep(0.2)

    for _ in range(warmup):
        try: func(*args)
        except: pass
        gc.collect()

    gc.collect()
    time.sleep(0.2)

    mon = Monitor()
    mon.start()
    t0 = time.perf_counter()

    success = True
    try:
        for _ in range(n_calls):
            func(*args)
    except Exception as e:
        success = False

    t1 = time.perf_counter()
    mem, cpu_max, cpu_avg = mon.stop()

    return {
        "runtime_ms": (t1 - t0) * 1000 / n_calls,
        "total_runtime_ms": (t1 - t0) * 1000,
        "memory_mb": mem,
        "cpu_max": cpu_max,
        "cpu_avg": cpu_avg,
        "success": success
    }


def main():
    print("=" * 85)
    print("   HIGH FREQUENCY BENCHMARK")
    print("   x1,000 | x5,000 | x10,000 | x50,000 calls")
    print("=" * 85)

    transform = get_transform()
    all_results = []

    # =================================================================
    # [1] image_preprocess - HIGH FREQUENCY
    # =================================================================
    print(f"\n{'='*85}")
    print("  [image_preprocess] - HIGH FREQUENCY TEST")
    print(f"{'='*85}")

    for size in IMAGE_SIZES:
        img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

        print(f"\n  Image size: {size[1]}x{size[0]}")
        print(f"  {'─'*85}")
        print(f"  {'Calls':>12} | {'A Runtime':>12} | {'B Runtime':>12} | {'A Mem':>10} | {'B Mem':>10} | {'Winner':>8}")
        print(f"  {'─'*85}")

        for n_calls in HIGH_CALLS:
            ra = benchmark(image_preprocess_a, img_np, transform, n_calls=n_calls, warmup=5)
            rb = benchmark(image_preprocess_b, img_np, transform, n_calls=n_calls, warmup=5)

            winner = "A" if ra["runtime_ms"] < rb["runtime_ms"] else "B"
            speedup = max(ra["runtime_ms"], rb["runtime_ms"]) / min(ra["runtime_ms"], rb["runtime_ms"])

            print(f"  {n_calls:>12,} | {ra['runtime_ms']:>10.6f}ms | {rb['runtime_ms']:>10.6f}ms | "
                  f"{ra['memory_mb']:>8.1f}MB | {rb['memory_mb']:>8.1f}MB | {winner} ({speedup:.2f}x)")

            all_results.append({
                "func": "image_preprocess", "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "a_runtime_ms": ra["runtime_ms"],
                "b_runtime_ms": rb["runtime_ms"],
                "a_memory_mb": ra["memory_mb"],
                "b_memory_mb": rb["memory_mb"],
                "a_cpu_max": ra["cpu_max"],
                "b_cpu_max": rb["cpu_max"],
                "winner": winner
            })

    # =================================================================
    # [2] preprocess_image - HIGH FREQUENCY
    # =================================================================
    print(f"\n{'='*85}")
    print("  [preprocess_image] - HIGH FREQUENCY TEST")
    print(f"{'='*85}")

    for size in IMAGE_SIZES:
        img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

        print(f"\n  Image size: {size[1]}x{size[0]}")
        print(f"  {'─'*85}")
        print(f"  {'Calls':>12} | {'A Runtime':>12} | {'B Runtime':>12} | {'A Mem':>10} | {'B Mem':>10} | {'Winner':>8}")
        print(f"  {'─'*85}")

        for n_calls in HIGH_CALLS:
            ra = benchmark(preprocess_image_a, img_np, n_calls=n_calls, warmup=5)
            rb = benchmark(preprocess_image_b, img_np, n_calls=n_calls, warmup=5)

            winner = "A" if ra["runtime_ms"] < rb["runtime_ms"] else "B"
            speedup = max(ra["runtime_ms"], rb["runtime_ms"]) / min(ra["runtime_ms"], rb["runtime_ms"])

            print(f"  {n_calls:>12,} | {ra['runtime_ms']:>10.6f}ms | {rb['runtime_ms']:>10.6f}ms | "
                  f"{ra['memory_mb']:>8.1f}MB | {rb['memory_mb']:>8.1f}MB | {winner} ({speedup:.2f}x)")

            all_results.append({
                "func": "preprocess_image", "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "a_runtime_ms": ra["runtime_ms"],
                "b_runtime_ms": rb["runtime_ms"],
                "a_memory_mb": ra["memory_mb"],
                "b_memory_mb": rb["memory_mb"],
                "a_cpu_max": ra["cpu_max"],
                "b_cpu_max": rb["cpu_max"],
                "winner": winner
            })

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 85)
    print("   SUMMARY - HIGH FREQUENCY")
    print("=" * 85)

    for func_name in ["image_preprocess", "preprocess_image"]:
        subset = [r for r in all_results if r["func"] == func_name]
        a_wins = sum(1 for r in subset if r["winner"] == "A")

        # By call frequency
        print(f"\n  [{func_name}]")

        for calls in HIGH_CALLS:
            freq_subset = [r for r in subset if r["calls"] == calls]
            avg_a = sum(r["a_runtime_ms"] for r in freq_subset) / len(freq_subset)
            avg_b = sum(r["b_runtime_ms"] for r in freq_subset) / len(freq_subset)
            a_w = sum(1 for r in freq_subset if r["winner"] == "A")
            print(f"    x{calls:>6,}: A wins {a_w}/{len(freq_subset)} (avg A={avg_a:.6f}ms, B={avg_b:.6f}ms)")

        all_a = sum(r["a_runtime_ms"] for r in subset) / len(subset)
        all_b = sum(r["b_runtime_ms"] for r in subset) / len(subset)
        print(f"    Overall: A {a_wins}/{len(subset)} wins, avg runtime A={all_a:.6f}ms, B={all_b:.6f}ms")

    # Overall
    a_wins_total = sum(1 for r in all_results if r["winner"] == "A")
    overall_a = sum(r["a_runtime_ms"] for r in all_results) / len(all_results)
    overall_b = sum(r["b_runtime_ms"] for r in all_results) / len(all_results)

    print(f"\n  [TOTAL]")
    print(f"    A wins: {a_wins_total}/{len(all_results)} ({a_wins_total/len(all_results)*100:.1f}%)")
    print(f"    Avg Runtime: A={overall_a:.6f}ms, B={overall_b:.6f}ms")
    print(f"    Speedup: {max(overall_a, overall_b)/min(overall_a, overall_b):.3f}x")

    # Save
    with open("high_freq_benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "a_wins": a_wins_total,
                "avg_runtime_a_ms": overall_a,
                "avg_runtime_b_ms": overall_b
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Report: high_freq_benchmark_report.json")
    print("=" * 85)


if __name__ == "__main__":
    main()