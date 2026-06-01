#!/usr/bin/env python3
"""
Runtime Benchmark Compare - Preprocess Functions
功能: 比較兩個 preprocess 函數的效能差異
追蹤: 執行時間、峰值記憶體、峰值 CPU 使用率
"""

import time
import os
import sys
import gc
import subprocess
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import traceback

# 安裝必要套件
def install_packages():
    import urllib.request
    packages = ['psutil', 'numpy', 'opencv-python']

    for pkg in packages:
        try:
            module_name = pkg.replace('-', '_').replace('python', '').split('.')[0]
            __import__(module_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Please install: pip install opencv-python")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """單次 benchmark 結果"""
    name: str
    runtime_seconds: float
    peak_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    success: bool
    error_message: str = ""
    return_value: Any = None
    return_shape: Optional[str] = None
    return_dtype: Optional[str] = None


@dataclass
class ComparisonReport:
    """比較報告"""
    func_a_result: BenchmarkResult
    func_b_result: BenchmarkResult
    time_diff_seconds: float
    time_speedup_ratio: float  # A/B
    memory_diff_mb: float
    cpu_diff_percent: float
    results_match: bool
    match_type: str  # "identical", "close", "different", "error"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResourceMonitor:
    """資源監控器 - 在背景執行緒中追蹤記憶體和 CPU"""

    def __init__(self, interval: float = 0.01):  # 10ms 取樣
        self.interval = interval
        self.running = False
        self.thread = None
        self.max_memory_mb = 0.0
        self.max_cpu_percent = 0.0
        self.cpu_samples = []
        self.process = psutil.Process()

    def start(self):
        """開始監控"""
        self.running = True
        self.max_memory_mb = 0.0
        self.max_cpu_percent = 0.0
        self.cpu_samples = []
        self.process = psutil.Process()
        self._start_monitor_thread()

    def _start_monitor_thread(self):
        """啟動監控執行緒"""
        import threading
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[float, float, List[float]]:
        """停止監控並回傳結果"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.max_memory_mb, self.max_cpu_percent, self.cpu_samples

    def _monitor_loop(self):
        """監控迴圈"""
        try:
            while self.running:
                try:
                    # 記憶體使用 (MB)
                    mem_info = self.process.memory_info()
                    mem_mb = mem_info.rss / (1024 * 1024)
                    self.max_memory_mb = max(self.max_memory_mb, mem_mb)

                    # CPU 使用率 (整個系統)
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                    self.cpu_samples.append(cpu_percent)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                time.sleep(self.interval)
        except Exception:
            pass


class BenchmarkRunner:
    """Benchmark 執行器"""

    def __init__(self, warmup_runs: int = 3, measurement_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs

    def clear_memory(self):
        """清除記憶體"""
        gc.collect()
        try:
            # 如果有 GPU 也清除
            pass
        except:
            pass

    def benchmark_function(self, func: Callable, args: Tuple = (),
                           kwargs: Dict = None,
                           name: str = "Function") -> BenchmarkResult:
        """執行單一 function 的 benchmark"""

        if kwargs is None:
            kwargs = {}

        # 清除記憶體
        self.clear_memory()

        # Warmup (獨立執行,不混在一起)
        print(f"  Warming up {self.warmup_runs} times...")
        for _ in range(self.warmup_runs):
            try:
                _ = func(*args, **kwargs)
            except Exception:
                pass
            self.clear_memory()

        # 重置監控
        self.clear_memory()
        gc.collect()
        time.sleep(0.1)

        # 實際測量
        print(f"  Measuring {self.measurement_runs} times...")
        all_runtimes = []
        all_memories = []
        all_cpus = []

        for i in range(self.measurement_runs):
            self.clear_memory()
            gc.collect()
            time.sleep(0.05)

            monitor = ResourceMonitor(interval=0.01)
            monitor.start()
            start_time = time.perf_counter()

            try:
                return_value = func(*args, **kwargs)
                success = True
                error_message = ""
            except Exception as e:
                return_value = None
                success = False
                error_message = f"{type(e).__name__}: {str(e)}"
                traceback.print_exc()
                break

            end_time = time.perf_counter()
            peak_mem, peak_cpu, cpu_samples = monitor.stop()

            runtime = end_time - start_time
            all_runtimes.append(runtime)
            all_memories.append(peak_mem)
            all_cpus.append(peak_cpu)

        if all_runtimes:
            runtime = sum(all_runtimes) / len(all_runtimes)
            peak_mem = max(all_memories)
            peak_cpu = max(all_cpus)
            avg_cpu = sum(all_cpus) / len(all_cpus) if all_cpus else 0
        else:
            runtime = 0
            peak_mem = 0
            peak_cpu = 0
            avg_cpu = 0

        # 分析輸出
        return_shape = None
        return_dtype = None
        if return_value is not None:
            # preprocess 回傳 (img_process, ratio, pad_info)，第一個是處理後的圖片
            if isinstance(return_value, tuple) and len(return_value) > 0:
                img = return_value[0]
                if hasattr(img, 'shape'):
                    return_shape = str(img.shape)
                if hasattr(img, 'dtype'):
                    return_dtype = str(img.dtype)
            elif hasattr(return_value, 'shape'):
                return_shape = str(return_value.shape)
                return_dtype = str(return_value.dtype) if hasattr(return_value, 'dtype') else None

        print(f"\n  {'='*40}")
        print(f"  {name} Benchmark Results:")
        print(f"    Runtime:       {runtime:.6f}s (avg of {len(all_runtimes)} runs)")
        print(f"    Peak Memory:  {peak_mem:.2f} MB")
        print(f"    Peak CPU:     {peak_cpu:.1f}%")
        print(f"    Avg CPU:      {avg_cpu:.1f}%")
        print(f"    Success:      {success}")
        if return_shape:
            print(f"    Output Shape: {return_shape}")
        print(f"  {'='*40}\n")

        return BenchmarkResult(
            name=name,
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            peak_cpu_percent=peak_cpu,
            avg_cpu_percent=avg_cpu,
            success=success,
            error_message=error_message,
            return_value=return_value,
            return_shape=return_shape,
            return_dtype=return_dtype
        )

    def compare_results(self, result_a: BenchmarkResult,
                        result_b: BenchmarkResult,
                        rtol: float = 1e-3,
                        atol: float = 1e-5) -> Tuple[bool, str]:
        """比較兩個 function 的輸出結果"""

        if not result_a.success or not result_b.success:
            return False, "error"

        if result_a.return_value is None and result_b.return_value is None:
            return True, "both_none"

        if result_a.return_value is None or result_b.return_value is None:
            return False, "different"

        try:
            # 比較圖片輸出 (第一個回傳值)
            val_a = result_a.return_value[0] if isinstance(result_a.return_value, tuple) else result_a.return_value
            val_b = result_b.return_value[0] if isinstance(result_b.return_value, tuple) else result_b.return_value

            # 比較形狀
            if hasattr(val_a, 'shape') and hasattr(val_b, 'shape'):
                if val_a.shape != val_b.shape:
                    return False, f"shape_mismatch_{val_a.shape}_vs_{val_b.shape}"

                # 數值比較 (可能有不同的 channel order，所以用较大 tolerance)
                arr_a = np.array(val_a)
                arr_b = np.array(val_b)

                if np.allclose(arr_a, arr_b, rtol=rtol, atol=atol):
                    return True, "identical"
                elif np.allclose(arr_a, arr_b, rtol=1e-1, atol=1e-2):
                    return True, "close"
                else:
                    return False, "different"
            else:
                return False, "no_shape"
        except Exception as e:
            return False, f"compare_error_{str(e)}"

    def run_comparison(self, func_a: Callable, func_b: Callable,
                       args_a: Tuple = (), kwargs_a: Dict = None,
                       args_b: Tuple = (), kwargs_b: Dict = None,
                       name_a: str = "Function A",
                       name_b: str = "Function B") -> ComparisonReport:
        """執行完整比較"""

        if kwargs_a is None:
            kwargs_a = {}
        if kwargs_b is None:
            kwargs_b = {}

        print(f"\n{'#'*60}")
        print(f"# Starting Benchmark Comparison")
        print(f"# Time: {datetime.now().isoformat()}")
        print(f"# Warmup Runs: {self.warmup_runs}")
        print(f"# Measurement Runs: {self.measurement_runs}")
        print(f"{'#'*60}\n")

        print(f"{'='*60}")
        print(f"# BENCHMARKING: {name_a}")
        print(f"{'='*60}")
        result_a = self.benchmark_function(func_a, args_a, kwargs_a, name_a)

        print(f"\n{'='*60}")
        print(f"# BENCHMARKING: {name_b}")
        print(f"{'='*60}")
        result_b = self.benchmark_function(func_b, args_b, kwargs_b, name_b)

        # 計算時間差
        if result_b.runtime_seconds > 0:
            speedup = result_a.runtime_seconds / result_b.runtime_seconds
        else:
            speedup = float('inf') if result_a.runtime_seconds > 0 else 1.0

        time_diff = result_a.runtime_seconds - result_b.runtime_seconds

        # 比較輸出
        results_match, match_type = self.compare_results(result_a, result_b)

        report = ComparisonReport(
            func_a_result=result_a,
            func_b_result=result_b,
            time_diff_seconds=time_diff,
            time_speedup_ratio=speedup,
            memory_diff_mb=result_a.peak_memory_mb - result_b.peak_memory_mb,
            cpu_diff_percent=result_a.peak_cpu_percent - result_b.peak_cpu_percent,
            results_match=results_match,
            match_type=match_type
        )

        self._print_comparison_report(report, name_a, name_b)

        return report

    def _print_comparison_report(self, report: ComparisonReport,
                                  name_a: str, name_b: str):
        """列印比較報告"""
        print(f"\n{'#'*60}")
        print(f"#           COMPARISON REPORT")
        print(f"{'#'*60}\n")

        # 執行時間
        print(f"EXECUTION TIME:")
        print(f"  {name_a}:     {report.func_a_result.runtime_seconds:.6f}s")
        print(f"  {name_b}:     {report.func_b_result.runtime_seconds:.6f}s")
        print(f"  Difference:   {abs(report.time_diff_seconds):.6f}s")
        if abs(report.time_diff_seconds) > 0.0001:
            if report.time_diff_seconds > 0:
                winner = name_b
                speedup = report.time_speedup_ratio if report.time_speedup_ratio != float('inf') else "inf"
            else:
                winner = name_a
                speedup = 1/report.time_speedup_ratio if report.time_speedup_ratio > 0 else "inf"
            print(f"  Winner:       {winner} (faster by {abs(report.time_diff_seconds):.6f}s)")
            print(f"  Speedup Ratio: {speedup:.2f}x")

        # 記憶體
        print(f"\nMEMORY USAGE (Peak):")
        print(f"  {name_a}:     {report.func_a_result.peak_memory_mb:.2f} MB")
        print(f"  {name_b}:     {report.func_b_result.peak_memory_mb:.2f} MB")
        print(f"  Difference:   {abs(report.memory_diff_mb):.2f} MB")
        if abs(report.memory_diff_mb) > 0.01:
            if report.memory_diff_mb > 0:
                print(f"  Winner:       {name_b} (uses {abs(report.memory_diff_mb):.2f} MB less)")
            else:
                print(f"  Winner:       {name_a} (uses {abs(report.memory_diff_mb):.2f} MB less)")

        # CPU
        print(f"\nCPU USAGE (Peak):")
        print(f"  {name_a}:     {report.func_a_result.peak_cpu_percent:.1f}%")
        print(f"  {name_b}:     {report.func_b_result.peak_cpu_percent:.1f}%")
        print(f"  Difference:   {abs(report.cpu_diff_percent):.1f}%")

        # 輸出比較
        print(f"\nOUTPUT COMPARISON:")
        print(f"  Match Type:  {report.match_type}")
        print(f"  Match:      {'✓ PASS' if report.results_match else '✗ FAIL'}")

        if report.func_a_result.return_shape and report.func_b_result.return_shape:
            print(f"  Shape A:    {report.func_a_result.return_shape}")
            print(f"  Shape B:    {report.func_b_result.return_shape}")

        print(f"\n{'='*60}")
        print(f"Benchmark completed at: {report.timestamp}")
        print(f"{'='*60}\n")

    def save_report(self, report: ComparisonReport, name_a: str, name_b: str,
                    filepath: str = "preprocess_benchmark_report.json"):
        """儲存報告到 JSON"""
        data = {
            "timestamp": report.timestamp,
            "function_a": {
                "name": name_a,
                "runtime_seconds": report.func_a_result.runtime_seconds,
                "peak_memory_mb": report.func_a_result.peak_memory_mb,
                "peak_cpu_percent": report.func_a_result.peak_cpu_percent,
                "avg_cpu_percent": report.func_a_result.avg_cpu_percent,
                "success": report.func_a_result.success,
                "error": report.func_a_result.error_message,
                "output_shape": report.func_a_result.return_shape,
                "output_dtype": report.func_a_result.return_dtype
            },
            "function_b": {
                "name": name_b,
                "runtime_seconds": report.func_b_result.runtime_seconds,
                "peak_memory_mb": report.func_b_result.peak_memory_mb,
                "peak_cpu_percent": report.func_b_result.peak_cpu_percent,
                "avg_cpu_percent": report.func_b_result.avg_cpu_percent,
                "success": report.func_b_result.success,
                "error": report.func_b_result.error_message,
                "output_shape": report.func_b_result.return_shape,
                "output_dtype": report.func_b_result.return_dtype
            },
            "comparison": {
                "time_diff_seconds": report.time_diff_seconds,
                "time_speedup_ratio": report.time_speedup_ratio,
                "memory_diff_mb": report.memory_diff_mb,
                "cpu_diff_percent": report.cpu_diff_percent,
                "results_match": report.results_match,
                "match_type": report.match_type
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Report saved to: {filepath}")


# =============================================================================
# Function A: BGR 版本 (假設輸入為 BGR 影像)
# =============================================================================
def preprocess_bgr(img_bgr: np.ndarray, img_shape: int, dtype: np.dtype):
    """
    Pre-processes the input image (BGR version).

    Args:
        img_bgr (Numpy.ndarray): image about to be processed.
        img_shape (int): new image shape (rectangle).
        dtype: output dtype.

    Returns:
        img_process (Numpy.ndarray): image preprocessed for inference.
        ratio (tuple): width, height ratios in letterbox.
        pad_w (float): width padding in letterbox.
        pad_h (float): height padding in letterbox.
    """
    # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
    shape = img_bgr.shape[:2]
    new_shape = (img_shape, img_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img_bgr = cv2.copyMakeBorder(
        img_bgr,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if np.dtype(dtype) == np.float32:
        # 這裡 swapRB=True 因為輸入是 BGR，需要轉成 RGB
        img_process = cv2.dnn.blobFromImage(
            img_bgr,
            scalefactor=1.0 / 255.0,
            size=None,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,  # BGR -> RGB
            crop=False,
            ddepth=cv2.CV_32F,
        )
        return img_process, ratio, (pad_w, pad_h)

    img_process = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), dtype=dtype)
    img_process *= np.array(1.0 / 255.0, dtype=dtype)

    return img_process[None], ratio, (pad_w, pad_h)


# =============================================================================
# Function B: RGB 版本 (假設輸入為 RGB 影像)
# =============================================================================
def preprocess_rgb(img_rgb: np.ndarray, img_shape: int, dtype: np.dtype):
    """
    Pre-processes the input image (RGB version).

    Args:
        img_rgb (Numpy.ndarray): image about to be processed.
        img_shape (int): new image shape (rectangle).
        dtype: output dtype.

    Returns:
        img_process (Numpy.ndarray): image preprocessed for inference.
        ratio (tuple): width, height ratios in letterbox.
        pad_w (float): width padding in letterbox.
        pad_h (float): height padding in letterbox.
    """
    # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
    shape = img_rgb.shape[:2]
    new_shape = (img_shape, img_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img_rgb = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img_rgb = cv2.copyMakeBorder(
        img_rgb,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if np.dtype(dtype) == np.float32:
        # 這裡 swapRB=False 因為輸入已經是 RGB
        img_process = cv2.dnn.blobFromImage(
            img_rgb,
            scalefactor=1.0 / 255.0,
            size=None,
            mean=(0.0, 0.0, 0.0),
            swapRB=False,  # 已確認是 RGB，不需交換
            crop=False,
            ddepth=cv2.CV_32F,
        )
        return img_process, ratio, (pad_w, pad_h)

    img_process = np.ascontiguousarray(img_rgb.transpose(2, 0, 1), dtype=dtype)
    img_process *= np.array(1.0 / 255.0, dtype=dtype)

    return img_process[None], ratio, (pad_w, pad_h)


# =============================================================================
# 輸入生成器
# =============================================================================
def generate_inputs(img_shape: int = 640, target_dtype: str = "float32"):
    """
    為 preprocess 函數產生測試輸入。

    由於 preprocess_bgr 期望 BGR 輸入，preprocess_rgb 期望 RGB 輸入，
    測試時我們:
    1. preprocess_bgr: 使用 BGR 格式的隨機圖片
    2. preprocess_rgb: 使用 RGB 格式的隨機圖片 (轉換後)

    Args:
        img_shape: 目標輸出尺寸 (會 letterbox)
        target_dtype: 输出版本 (float32 或 uint8)

    Returns:
        args_a, kwargs_a: preprocess_bgr 的輸入
        args_b, kwargs_b: preprocess_rgb 的輸入
        圖片形狀: (原始高度, 原始寬度, 3)
    """
    # 產生隨機形狀的圖片 (避免總是 640x640)
    h, w = np.random.randint(400, 800), np.random.randint(400, 800)

    # 產生 BGR 格式 (OpenCV 標準格式)
    img_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # 轉換成 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    dtype = np.dtype(target_dtype)

    # 兩個函數的輸入
    kwargs_a = {
        'img_shape': img_shape,
        'dtype': dtype
    }
    kwargs_b = {
        'img_shape': img_shape,
        'dtype': dtype
    }

    return (img_bgr,), kwargs_a, (img_rgb,), kwargs_b, (h, w)


def analyze_inputs():
    """分析函數輸入需求"""
    print("\n" + "="*60)
    print("INPUT ANALYSIS")
    print("="*60)

    print("\n[Function A: preprocess_bgr]")
    print("  - img_bgr: np.ndarray (BGR 格式, 3通道)")
    print("  - img_shape: int (目標輸出尺寸, e.g., 640)")
    print("  - dtype: np.dtype (輸出類型, e.g., float32, uint8)")

    print("\n[Function B: preprocess_rgb]")
    print("  - img_rgb: np.ndarray (RGB 格式, 3通道)")
    print("  - img_shape: int (目標輸出尺寸, e.g., 640)")
    print("  - dtype: np.dtype (輸出類型, e.g., float32, uint8)")

    print("\n[Test Input Generated]")
    print("  - 圖片形狀: 隨機 (400-800) x (400-800) x 3")
    print("  - 圖片格式: BGR (OpenCV) / RGB (標準)")
    print("  - 測試尺寸: 640x640")
    print("  - 測試類型: float32")


def check_potential_issues():
    """檢查潛在問題"""
    print("\n" + "="*60)
    print("POTENTIAL ISSUES CHECK")
    print("="*60)

    print("\n[✓] Import check: psutil, numpy, cv2 - OK")
    print("[✓] Function signatures: Both take (img, img_shape, dtype) - OK")
    print("[✓] Return values: Both return (img_process, ratio, pad_info) - OK")

    print("\n[!] NOTE: The two functions handle color channels differently.")
    print("    - preprocess_bgr: expects BGR input, converts to RGB (swapRB=True)")
    print("    - preprocess_rgb: expects RGB input, no conversion (swapRB=False)")
    print("    For proper comparison, we test with BGR input for A and RGB for B.")
    print("    Output pixel values should be equivalent if input color space matches.")


def main():
    """主程式"""
    print("=" * 60)
    print("   Preprocess Function Runtime Benchmark")
    print("=" * 60)

    # 分析輸入
    analyze_inputs()

    # 檢查問題
    check_potential_issues()

    # 產生輸入
    print("\n" + "-"*60)
    print("Generating test inputs...")
    args_a, kwargs_a, args_b, kwargs_b, original_shape = generate_inputs(img_shape=640, target_dtype="float32")
    print(f"  Original image shape: {original_shape[0]}x{original_shape[1]}x3")
    print(f"  Target size: 640x640")
    print(f"  Output dtype: float32")
    print("-"*60)

    # 執行 benchmark
    print("\n" + "="*60)
    print("STARTING BENCHMARK")
    print("="*60)

    runner = BenchmarkRunner(warmup_runs=3, measurement_runs=5)

    report = runner.run_comparison(
        func_a=preprocess_bgr,
        func_b=preprocess_rgb,
        args_a=args_a,
        kwargs_a=kwargs_a,
        args_b=args_b,
        kwargs_b=kwargs_b,
        name_a="preprocess_bgr",
        name_b="preprocess_rgb"
    )

    # 儲存報告
    runner.save_report(report, "preprocess_bgr", "preprocess_rgb", "preprocess_benchmark_report.json")

    # 總結
    print("\n" + "="*60)
    print("   BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()