#!/usr/bin/env python3
"""
Benchmark: Original PIL-based transforms vs NumPy-compatible transforms
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
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from groundingdino.transforms_np import (
    crop, hflip, resize, pad, to_tensor, to_pil_image, to_numpy,
    RandomHorizontalFlip, RandomResize, ToTensor, Normalize, Compose,
    get_image_size, is_numpy_image
)

# =============================================================================
# Benchmark Helper
# =============================================================================
class ResourceMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.running = False
        self.thread = None
        self.max_mem = 0.0
        self.max_cpu = 0.0
        self.process = None

    def start(self):
        self.running = True
        self.max_mem = 0.0
        self.max_cpu = 0.0
        self.process = psutil.Process()
        import threading
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.running:
                try:
                    self.max_mem = max(self.max_mem, self.process.memory_info().rss / (1024*1024))
                    self.max_cpu = max(self.max_cpu, psutil.cpu_percent(interval=None))
                except: pass
                time.sleep(self.interval)
        except: pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.max_mem, self.max_cpu


def benchmark(func, image, target, n_calls, warmup=2):
    gc.collect()
    time.sleep(0.1)

    # Warmup
    for _ in range(warmup):
        try:
            _ = func(image, target)
        except Exception as e:
            print(f"Warmup error: {e}")
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
            _ = func(image, target)
    except Exception as e:
        success = False
        print(f"Error: {e}")

    t1 = time.perf_counter()
    mem, cpu = mon.stop()

    return (t1 - t0) * 1000 / n_calls, mem, cpu, success


# =============================================================================
# Original PIL-based transforms (for comparison)
# =============================================================================
def crop_original(image, target, region):
    i, j, h, w = region
    cropped = F.crop(image, *region)
    target = target.copy()
    target["size"] = torch.tensor([h, w])
    return cropped, target


def hflip_original(image, target):
    flipped = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    return flipped, target


def resize_original(image, target, size, max_size=None):
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            w, h = image_size
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            return (oh, ow)

    image_size = image.size
    size_wh = get_size(image_size, size, max_size)
    rescaled = F.resize(image, size_wh)

    if target is None:
        return rescaled, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled.size, image_size))
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled = boxes * torch.as_tensor([ratios[0], ratios[1], ratios[0], ratios[1]])
        target["boxes"] = scaled
    if "area" in target:
        target["area"] = target["area"] * (ratios[0] * ratios[1])
    target["size"] = torch.tensor(list(rescaled.size)[::-1])
    return rescaled, target


# =============================================================================
# Tests
# =============================================================================
IMAGE_SIZES = [(480, 640), (720, 1280), (1080, 1920)]
CALLS = [1, 10, 100]


def main():
    print("=" * 70)
    print("   Transforms Benchmark: PIL vs Numpy-Compatible")
    print("=" * 70)

    all_results = []

    for size in IMAGE_SIZES:
        for n_calls in CALLS:
            print(f"\n{'─'*70}")
            print(f"  Size: {size[1]}x{size[0]}, Calls: {n_calls}")
            print(f"{'─'*70}")

            # Create test data
            # NumPy (for new version)
            img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            # PIL (for original)
            img_pil = Image.fromarray(img_np)

            # Target
            target = {
                "boxes": torch.tensor([
                    [size[1]//4, size[0]//4, size[1]//2, size[0]//2],
                    [size[1]//8, size[0]//8, size[1]//4, size[0]//4]
                ], dtype=torch.float32),
                "labels": torch.tensor([1, 2]),
                "area": torch.tensor([10000.0, 5000.0])
            }

            # ========== Test 1: RandomHorizontalFlip ==========
            transform_new = RandomHorizontalFlip(p=0.5)
            transform_pil = T.RandomHorizontalFlip(p=0.5)

            t_fl = benchmark(transform_new, img_np, target.copy(), n_calls)
            t_fl_pil = benchmark(lambda i, t: hflip_original(i, t), img_pil, target.copy(), n_calls)

            print(f"  [hflip]")
            print(f"    Numpy:   {t_fl[0]:.3f} ms  {'✓' if t_fl[3] else '✗'}")
            print(f"    PIL:     {t_fl_pil[0]:.3f} ms  {'✓' if t_fl_pil[3] else '✗'}")

            all_results.append({
                "test": "hflip",
                "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "numpy_ms": t_fl[0],
                "pil_ms": t_fl_pil[0],
                "winner": "numpy" if t_fl[0] < t_fl_pil[0] else "pil",
                "diff_pct": abs(t_fl[0] - t_fl_pil[0]) / max(t_fl[0], t_fl_pil[0]) * 100
            })

            # ========== Test 2: Resize ==========
            transform_resize_new = RandomResize([800], max_size=1333)

            t_rz = benchmark(transform_resize_new, img_np, target.copy(), n_calls)
            # For PIL comparison, use the original resize function
            t_rz_pil = benchmark(lambda i, t: resize_original(i, t, 800, 1333), img_pil, target.copy(), n_calls)

            print(f"  [resize]")
            print(f"    Numpy:   {t_rz[0]:.3f} ms  {'✓' if t_rz[3] else '✗'}")
            print(f"    PIL:     {t_rz_pil[0]:.3f} ms  {'✓' if t_rz_pil[3] else '✗'}")

            all_results.append({
                "test": "resize",
                "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "numpy_ms": t_rz[0],
                "pil_ms": t_rz_pil[0],
                "winner": "numpy" if t_rz[0] < t_rz_pil[0] else "pil",
                "diff_pct": abs(t_rz[0] - t_rz_pil[0]) / max(t_rz[0], t_rz_pil[0]) * 100
            })

            # ========== Test 3: ToTensor + Normalize ==========
            transforms_new = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            transforms_pil = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            t_tn = benchmark(transforms_new, img_np, target.copy(), n_calls)
            t_tn_pil = benchmark(transforms_pil, img_pil, target.copy(), n_calls)

            print(f"  [ToTensor+Normalize]")
            print(f"    Numpy:   {t_tn[0]:.3f} ms  {'✓' if t_tn[3] else '✗'}")
            print(f"    PIL:     {t_tn_pil[0]:.3f} ms  {'✓' if t_tn_pil[3] else '✗'}")

            all_results.append({
                "test": "ToTensor+Normalize",
                "size": f"{size[1]}x{size[0]}",
                "calls": n_calls,
                "numpy_ms": t_tn[0],
                "pil_ms": t_tn_pil[0],
                "winner": "numpy" if t_tn[0] < t_tn_pil[0] else "pil",
                "diff_pct": abs(t_tn[0] - t_tn_pil[0]) / max(t_tn[0], t_tn_pil[0]) * 100
            })

            gc.collect()
            time.sleep(0.1)

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)

    numpy_wins = sum(1 for r in all_results if r["winner"] == "numpy")
    pil_wins = sum(1 for r in all_results if r["winner"] == "pil")
    total = len(all_results)

    avg_numpy = sum(r["numpy_ms"] for r in all_results) / total
    avg_pil = sum(r["pil_ms"] for r in all_results) / total

    print(f"\n  Overall Win Count:")
    print(f"    NumPy:    {numpy_wins}/{total}")
    print(f"    PIL:      {pil_wins}/{total}")

    print(f"\n  Average Runtime:")
    print(f"    NumPy:    {avg_numpy:.3f} ms")
    print(f"    PIL:      {avg_pil:.3f} ms")
    print(f"    Winner:   {'NumPy' if avg_numpy < avg_pil else 'PIL'}  (diff: {abs(avg_numpy-avg_pil)/max(avg_numpy,avg_pil)*100:.1f}%)")

    # By test type
    print(f"\n  By Transform Type:")
    for test_type in ["hflip", "resize", "ToTensor+Normalize"]:
        subset = [r for r in all_results if r["test"] == test_type]
        if subset:
            nw = sum(1 for r in subset if r["winner"] == "numpy")
            avg_n = sum(r["numpy_ms"] for r in subset) / len(subset)
            avg_p = sum(r["pil_ms"] for r in subset) / len(subset)
            print(f"    {test_type}: NumPy wins {nw}/{len(subset)} (avg {avg_n:.3f}ms vs {avg_p:.3f}ms)")

    # Save
    with open("transforms_benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "numpy_wins": numpy_wins,
                "pil_wins": pil_wins,
                "avg_numpy_ms": avg_numpy,
                "avg_pil_ms": avg_pil
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: transforms_benchmark_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()