#!/usr/bin/env python3
"""
Runtime Benchmark - Face Recognition (preprocess + cosine_similarity + max)
"""

import time
import gc
import sys
import subprocess
import json
from datetime import datetime
from typing import Tuple
from dataclasses import dataclass

def install_packages():
    for pkg in ['psutil', 'numpy']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import numpy as np


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
# 版本 A: 原始版本
# =============================================================================
class FaceProcessorA:
    @staticmethod
    def preprocess(face_image: np.ndarray) -> np.ndarray:
        """Scale input image to [-1, 1]"""
        return ((face_image / 255. - 0.5) / 0.5)[None,...]

    @staticmethod
    def cosine_similarity(session_output: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity"""
        return np.dot(session_output, np.transpose(features))

    @staticmethod
    def _get_max_similarity(similarity: np.ndarray) -> Tuple[float, int]:
        """Return max value and index"""
        return np.max(similarity), np.argmax(similarity)


# =============================================================================
# 版本 B: 優化版本
# =============================================================================
class FaceProcessorB:
    @staticmethod
    def preprocess(face_image: np.ndarray) -> np.ndarray:
        """Scale input image to [-1, 1]"""
        face_image = face_image.astype(np.float32, copy=False)
        return ((face_image / 255. - 0.5) / 0.5)[None,...]

    @staticmethod
    def cosine_similarity(session_output: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity"""
        return session_output @ features.T

    @staticmethod
    def _get_max_similarity(similarity: np.ndarray) -> Tuple[float, int]:
        """Return max value and index"""
        similarity = np.ascontiguousarray(similarity, dtype=np.float32)
        return similarity.max(), similarity.argmax()


@dataclass
class BenchResult:
    name: str
    preprocess_ms: float
    cosine_ms: float
    max_ms: float
    total_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool


def benchmark_full_pipeline(processor, face_image, features, n_calls, warmup=2):
    """測試完整 pipeline: preprocess -> cosine -> max"""
    gc.collect()
    time.sleep(0.1)

    # Warmup
    for _ in range(warmup):
        _ = processor.preprocess(face_image)
        feat = processor.preprocess(face_image)
        sim = processor.cosine_similarity(feat, features)
        _ = processor._get_max_similarity(sim)
        gc.collect()

    gc.collect()
    time.sleep(0.1)

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()

    success = True
    try:
        for _ in range(n_calls):
            feat = processor.preprocess(face_image)
            sim = processor.cosine_similarity(feat, features)
            max_val, max_idx = processor._get_max_similarity(sim)
    except Exception as e:
        success = False
        print(f"Error: {e}")

    t1 = time.perf_counter()
    mem, cpu, _ = mon.stop()

    total_ms = (t1 - t0) * 1000 / n_calls

    return BenchResult(
        name=processor.__name__,
        preprocess_ms=0,
        cosine_ms=0,
        max_ms=0,
        total_ms=total_ms,
        memory_mb=mem,
        cpu_percent=cpu,
        success=success
    )


def benchmark_individual(processor, face_image, session_output, features, n_calls, warmup=2):
    """個別測量每個函數"""
    gc.collect()
    time.sleep(0.1)

    results = {}

    # Preprocess (輸出會是 (1, H, W, C))
    for _ in range(warmup):
        _ = processor.preprocess(face_image)
    gc.collect()

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = processor.preprocess(face_image)
    t1 = time.perf_counter()
    mem, _, _ = mon.stop()
    results['preprocess_ms'] = (t1 - t0) * 1000 / n_calls

    # Cosine (使用 session_output，不是 preprocess 輸出)
    for _ in range(warmup):
        _ = processor.cosine_similarity(session_output, features)
    gc.collect()

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = processor.cosine_similarity(session_output, features)
    t1 = time.perf_counter()
    mem, _, _ = mon.stop()
    results['cosine_ms'] = (t1 - t0) * 1000 / n_calls

    # Max
    sim = processor.cosine_similarity(session_output, features)
    for _ in range(warmup):
        _ = processor._get_max_similarity(sim)
    gc.collect()

    mon = ResourceMonitor(0.01)
    mon.start()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = processor._get_max_similarity(sim)
    t1 = time.perf_counter()
    mem, _, _ = mon.stop()
    results['max_ms'] = (t1 - t0) * 1000 / n_calls

    return results


# 測試配置
FACE_SIZES = [(112, 112, 3), (224, 224, 3), (256, 256, 3)]  # 常見臉部圖片尺寸
FEATURE_SIZES = [1000, 10000, 100000]  # 特徵庫大小
CALLS = [1, 100, 1000]


def main():
    print("=" * 70)
    print("   Face Recognition Functions Benchmark")
    print("   A: Original  |  B: Optimized")
    print("=" * 70)

    all_results = []

    for face_size in FACE_SIZES:
        for feat_size in FEATURE_SIZES:
            for n_calls in CALLS:
                print(f"\n{'─'*70}")
                print(f"  Face: {face_size[0]}x{face_size[1]}, Features: {feat_size}, Calls: {n_calls}")
                print(f"{'─'*70}")

                # 產生測試資料
                face_image = np.random.randint(0, 256, face_size, dtype=np.uint8)

                # session_output 是模型輸出 (1, 512)，不是從 face_image 來的
                session_output = np.random.randn(1, 512).astype(np.float32)
                features = np.random.randn(feat_size, 512).astype(np.float32)

                # preprocess 的輸出形狀會是 (1, H, W, C)
                # 但我們要測試的是 preprocess 本身的效能，不是完整 pipeline

                # 版本 A
                ra_ind = benchmark_individual(FaceProcessorA, face_image, session_output, features, n_calls)
                print(f"    A preprocess: {ra_ind['preprocess_ms']:.6f} ms")
                print(f"    A cosine:     {ra_ind['cosine_ms']:.6f} ms")
                print(f"    A max:        {ra_ind['max_ms']:.6f} ms")

                # 版本 B
                rb_ind = benchmark_individual(FaceProcessorB, face_image, session_output, features, n_calls)
                print(f"    B preprocess: {rb_ind['preprocess_ms']:.6f} ms")
                print(f"    B cosine:     {rb_ind['cosine_ms']:.6f} ms")
                print(f"    B max:        {rb_ind['max_ms']:.6f} ms")

                # 差異
                total_a = sum(ra_ind.values())
                total_b = sum(rb_ind.values())
                winner = "B" if total_b < total_a else "A"
                diff_pct = abs(total_a - total_b) / total_a * 100

                print(f"    ─────────────────────────────────────────")
                print(f"    Total A: {total_a:.6f} ms  |  Total B: {total_b:.6f} ms")
                print(f"    Winner: {winner}  (diff: {diff_pct:.1f}%)")

                all_results.append({
                    "face_size": f"{face_size[0]}x{face_size[1]}",
                    "feature_size": feat_size,
                    "calls": n_calls,
                    "a_preprocess_ms": ra_ind['preprocess_ms'],
                    "a_cosine_ms": ra_ind['cosine_ms'],
                    "a_max_ms": ra_ind['max_ms'],
                    "a_total_ms": total_a,
                    "b_preprocess_ms": rb_ind['preprocess_ms'],
                    "b_cosine_ms": rb_ind['cosine_ms'],
                    "b_max_ms": rb_ind['max_ms'],
                    "b_total_ms": total_b,
                    "winner": winner,
                    "diff_pct": diff_pct
                })

                gc.collect()
                time.sleep(0.1)

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)

    a_wins = sum(1 for r in all_results if r["winner"] == "A")
    b_wins = sum(1 for r in all_results if r["winner"] == "B")

    avg_a = sum(r["a_total_ms"] for r in all_results) / len(all_results)
    avg_b = sum(r["b_total_ms"] for r in all_results) / len(all_results)

    print(f"\n  A wins: {a_wins}/{len(all_results)}")
    print(f"  B wins: {b_wins}/{len(all_results)}")
    print(f"\n  Avg Runtime:")
    print(f"    A: {avg_a:.6f} ms")
    print(f"    B: {avg_b:.6f} ms")
    print(f"\n  Overall: {'B' if avg_b < avg_a else 'A'} is faster by {abs(avg_a - avg_b) / max(avg_a, avg_b) * 100:.1f}%")

    # 分項 summary
    print(f"\n  By Function:")
    avg_a_pre = sum(r["a_preprocess_ms"] for r in all_results) / len(all_results)
    avg_b_pre = sum(r["b_preprocess_ms"] for r in all_results) / len(all_results)
    avg_a_cos = sum(r["a_cosine_ms"] for r in all_results) / len(all_results)
    avg_b_cos = sum(r["b_cosine_ms"] for r in all_results) / len(all_results)
    avg_a_max = sum(r["a_max_ms"] for r in all_results) / len(all_results)
    avg_b_max = sum(r["b_max_ms"] for r in all_results) / len(all_results)

    print(f"    preprocess:  A={avg_a_pre:.6f}ms  B={avg_b_pre:.6f}ms  {'B' if avg_b_pre < avg_a_pre else 'A'} wins")
    print(f"    cosine:      A={avg_a_cos:.6f}ms  B={avg_b_cos:.6f}ms  {'B' if avg_b_cos < avg_a_cos else 'A'} wins")
    print(f"    max:         A={avg_a_max:.6f}ms  B={avg_b_max:.6f}ms  {'B' if avg_b_max < avg_a_max else 'A'} wins")

    # Save
    with open("face_recognition_benchmark_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "avg_a_ms": avg_a,
                "avg_b_ms": avg_b,
                "avg_preprocess_a": avg_a_pre,
                "avg_preprocess_b": avg_b_pre,
                "avg_cosine_a": avg_a_cos,
                "avg_cosine_b": avg_b_cos,
                "avg_max_a": avg_a_max,
                "avg_max_b": avg_b_max
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: face_recognition_benchmark_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()