#!/usr/bin/env python3
"""
Benchmark: load_image, image_preprocess, preprocess_image
完整收集 Runtime + Memory + CPU 資料
"""

import time
import gc
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

def install_packages():
    for pkg in ['psutil', 'numpy', 'opencv-python', 'pillow', 'torch', 'torchvision']:
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


# =============================================================================
# Transform & Functions
# =============================================================================
def get_transform():
    return T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# --- Version A: PIL-based ---
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


# --- Version B: OpenCV + dtype check ---
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
                time.sleep(0.01)
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
CALLS = [1, 10, 100, 1000]
TEST_IMG_DIR = Path("test_images")


def create_test_images():
    TEST_IMG_DIR.mkdir(exist_ok=True)
    for size in IMAGE_SIZES:
        path = TEST_IMG_DIR / f"test_{size[1]}x{size[0]}.jpg"
        if not path.exists():
            img = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            cv2.imwrite(str(path), img)
    return list(TEST_IMG_DIR.glob("*.jpg"))


def benchmark(func, *args, n_calls=10, warmup=2):
    gc.collect()
    time.sleep(0.1)

    for _ in range(warmup):
        try: func(*args)
        except: pass
        gc.collect()

    gc.collect()
    time.sleep(0.1)

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
        "memory_mb": mem,
        "cpu_max": cpu_max,
        "cpu_avg": cpu_avg,
        "success": success
    }


def main():
    print("=" * 80)
    print("   Image Processing Functions - Full Benchmark")
    print("   Runtime + Memory + CPU")
    print("=" * 80)

    test_images = create_test_images()
    print(f"\nTest images: {len(test_images)}")
    transform = get_transform()

    all_results = []

    # =================================================================
    # [1] load_image
    # =================================================================
    print(f"\n{'='*80}")
    print("  [1] load_image")
    print(f"{'='*80}")

    for size in IMAGE_SIZES:
        for n_calls in CALLS[:3]:
            matching = [img for img in test_images if f"{size[1]}x{size[0]}" in img.name]
            if not matching: continue

            print(f"\n  {size[1]}x{size[0]} x{n_calls}")

            ra = benchmark(load_image_a, str(matching[0]), n_calls=n_calls)
            rb = benchmark(load_image_b, str(matching[0]), n_calls=n_calls)

            winner = "A" if ra["runtime_ms"] < rb["runtime_ms"] else "B"
            diff = abs(ra["runtime_ms"] - rb["runtime_ms"]) / max(ra["runtime_ms"], rb["runtime_ms"]) * 100

            print(f"    A (PIL): Runtime={ra['runtime_ms']:.4f}ms | Mem={ra['memory_mb']:.1f}MB | CPU={ra['cpu_max']:.1f}%  {'✓' if ra['success'] else '✗'}")
            print(f"    B (cv2): Runtime={rb['runtime_ms']:.4f}ms | Mem={rb['memory_mb']:.1f}MB | CPU={rb['cpu_max']:.1f}%  {'✓' if rb['success'] else '✗'}")
            print(f"    Winner: {winner} ({diff:.1f}%)")

            all_results.append({
                "func": "load_image", "size": f"{size[1]}x{size[0]}", "calls": n_calls,
                "a_runtime": ra["runtime_ms"], "b_runtime": rb["runtime_ms"],
                "a_memory": ra["memory_mb"], "b_memory": rb["memory_mb"],
                "a_cpu": ra["cpu_max"], "b_cpu": rb["cpu_max"],
                "winner": winner
            })

    # =================================================================
    # [2] image_preprocess
    # =================================================================
    print(f"\n{'='*80}")
    print("  [2] image_preprocess")
    print(f"{'='*80}")

    for size in IMAGE_SIZES:
        for n_calls in CALLS:
            print(f"\n  {size[1]}x{size[0]} x{n_calls}")

            img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

            ra = benchmark(image_preprocess_a, img_np, transform, n_calls=n_calls)
            rb = benchmark(image_preprocess_b, img_np, transform, n_calls=n_calls)

            winner = "A" if ra["runtime_ms"] < rb["runtime_ms"] else "B"
            diff = abs(ra["runtime_ms"] - rb["runtime_ms"]) / max(ra["runtime_ms"], rb["runtime_ms"]) * 100

            print(f"    A (PIL): Runtime={ra['runtime_ms']:.4f}ms | Mem={ra['memory_mb']:.1f}MB | CPU={ra['cpu_max']:.1f}%  {'✓' if ra['success'] else '✗'}")
            print(f"    B (cv2): Runtime={rb['runtime_ms']:.4f}ms | Mem={rb['memory_mb']:.1f}MB | CPU={rb['cpu_max']:.1f}%  {'✓' if rb['success'] else '✗'}")
            print(f"    Winner: {winner} ({diff:.1f}%)")

            all_results.append({
                "func": "image_preprocess", "size": f"{size[1]}x{size[0]}", "calls": n_calls,
                "a_runtime": ra["runtime_ms"], "b_runtime": rb["runtime_ms"],
                "a_memory": ra["memory_mb"], "b_memory": rb["memory_mb"],
                "a_cpu": ra["cpu_max"], "b_cpu": rb["cpu_max"],
                "winner": winner
            })

    # =================================================================
    # [3] preprocess_image
    # =================================================================
    print(f"\n{'='*80}")
    print("  [3] preprocess_image")
    print(f"{'='*80}")

    for size in IMAGE_SIZES:
        for n_calls in CALLS:
            print(f"\n  {size[1]}x{size[0]} x{n_calls}")

            img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

            ra = benchmark(preprocess_image_a, img_np, n_calls=n_calls)
            rb = benchmark(preprocess_image_b, img_np, n_calls=n_calls)

            winner = "A" if ra["runtime_ms"] < rb["runtime_ms"] else "B"
            diff = abs(ra["runtime_ms"] - rb["runtime_ms"]) / max(ra["runtime_ms"], rb["runtime_ms"]) * 100

            print(f"    A (PIL): Runtime={ra['runtime_ms']:.4f}ms | Mem={ra['memory_mb']:.1f}MB | CPU={ra['cpu_max']:.1f}%  {'✓' if ra['success'] else '✗'}")
            print(f"    B (cv2): Runtime={rb['runtime_ms']:.4f}ms | Mem={rb['memory_mb']:.1f}MB | CPU={rb['cpu_max']:.1f}%  {'✓' if rb['success'] else '✗'}")
            print(f"    Winner: {winner} ({diff:.1f}%)")

            all_results.append({
                "func": "preprocess_image", "size": f"{size[1]}x{size[0]}", "calls": n_calls,
                "a_runtime": ra["runtime_ms"], "b_runtime": rb["runtime_ms"],
                "a_memory": ra["memory_mb"], "b_memory": rb["memory_mb"],
                "a_cpu": ra["cpu_max"], "b_cpu": rb["cpu_max"],
                "winner": winner
            })

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)

    # By function
    for func_name in ["load_image", "image_preprocess", "preprocess_image"]:
        subset = [r for r in all_results if r["func"] == func_name]
        a_wins = sum(1 for r in subset if r["winner"] == "A")

        avg_a_runtime = sum(r["a_runtime"] for r in subset) / len(subset)
        avg_b_runtime = sum(r["b_runtime"] for r in subset) / len(subset)
        avg_a_mem = sum(r["a_memory"] for r in subset) / len(subset)
        avg_b_mem = sum(r["b_memory"] for r in subset) / len(subset)
        avg_a_cpu = sum(r["a_cpu"] for r in subset) / len(subset)
        avg_b_cpu = sum(r["b_cpu"] for r in subset) / len(subset)

        print(f"\n  [{func_name}]")
        print(f"    Win: A {a_wins}/{len(subset)}  B {len(subset)-a_wins}/{len(subset)}")
        print(f"    Runtime:  A={avg_a_runtime:.4f}ms  B={avg_b_runtime:.4f}ms  (winner: {'A' if avg_a_runtime < avg_b_runtime else 'B'})")
        print(f"    Memory:    A={avg_a_mem:.1f}MB    B={avg_b_mem:.1f}MB    (winner: {'A' if avg_a_mem < avg_b_mem else 'B'})")
        print(f"    CPU:       A={avg_a_cpu:.1f}%     B={avg_b_cpu:.1f}%     (winner: {'A' if avg_a_cpu < avg_b_cpu else 'B'})")

    # Overall
    a_wins_total = sum(1 for r in all_results if r["winner"] == "A")
    overall_a_runtime = sum(r["a_runtime"] for r in all_results) / len(all_results)
    overall_b_runtime = sum(r["b_runtime"] for r in all_results) / len(all_results)
    overall_a_mem = sum(r["a_memory"] for r in all_results) / len(all_results)
    overall_b_mem = sum(r["b_memory"] for r in all_results) / len(all_results)
    overall_a_cpu = sum(r["a_cpu"] for r in all_results) / len(all_results)
    overall_b_cpu = sum(r["b_cpu"] for r in all_results) / len(all_results)

    print(f"\n  [OVERALL]")
    print(f"    Win: A {a_wins_total}/{len(all_results)}  B {len(all_results)-a_wins_total}/{len(all_results)}")
    print(f"    Runtime:  A={overall_a_runtime:.4f}ms  B={overall_b_runtime:.4f}ms  (winner: {'A' if overall_a_runtime < overall_b_runtime else 'B'})")
    print(f"    Memory:   A={overall_a_mem:.1f}MB    B={overall_b_mem:.1f}MB    (winner: {'A' if overall_a_mem < overall_b_mem else 'B'})")
    print(f"    CPU:      A={overall_a_cpu:.1f}%     B={overall_b_cpu:.1f}%     (winner: {'A' if overall_a_cpu < overall_b_cpu else 'B'})")

    # Save
    with open("image_functions_full_benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "a_wins": a_wins_total,
                "runtime_avg_a": overall_a_runtime,
                "runtime_avg_b": overall_b_runtime,
                "memory_avg_a": overall_a_mem,
                "memory_avg_b": overall_b_mem,
                "cpu_avg_a": overall_a_cpu,
                "cpu_avg_b": overall_b_cpu
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: image_functions_full_benchmark_report.json")
    print("=" * 80)


if __name__ == "__main__":
    main()