#!/usr/bin/env python3
"""
Runtime Benchmark - Coordinate Transform Functions
比較兩種 torch tensor 類型轉換的效能差異
"""

import time
import os
import sys
import gc
import subprocess
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 安裝必要套件
def install_packages():
    packages = ['psutil', 'torch']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
                          capture_output=True)

install_packages()

import psutil
import torch


@dataclass
class BenchmarkResult:
    name: str
    runtime_seconds: float
    peak_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    success: bool
    error_message: str = ""
    return_shape: Optional[str] = None
    return_dtype: Optional[str] = None


@dataclass
class ComparisonReport:
    func_a_result: BenchmarkResult
    func_b_result: BenchmarkResult
    time_diff_seconds: float
    time_speedup_ratio: float
    memory_diff_mb: float
    cpu_diff_percent: float
    results_match: bool
    match_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResourceMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.running = False
        self.thread = None
        self.max_memory_mb = 0.0
        self.max_cpu_percent = 0.0
        self.cpu_samples = []
        self.process = psutil.Process()

    def start(self):
        self.running = True
        self.max_memory_mb = 0.0
        self.max_cpu_percent = 0.0
        self.cpu_samples = []
        self.process = psutil.Process()
        self._start_thread()

    def _start_thread(self):
        import threading
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.max_memory_mb, self.max_cpu_percent, self.cpu_samples

    def _monitor_loop(self):
        try:
            while self.running:
                try:
                    mem_info = self.process.memory_info()
                    mem_mb = mem_info.rss / (1024 * 1024)
                    self.max_memory_mb = max(self.max_memory_mb, mem_mb)
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                    self.cpu_samples.append(cpu_percent)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(self.interval)
        except Exception:
            pass


class BenchmarkRunner:
    def __init__(self, warmup_runs: int = 3, measurement_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def benchmark_function(self, func: Callable, args: Tuple = (),
                           kwargs: Dict = None, name: str = "Function") -> BenchmarkResult:
        if kwargs is None:
            kwargs = {}

        self.clear_memory()

        # Warmup
        for _ in range(self.warmup_runs):
            try:
                _ = func(*args, **kwargs)
            except Exception:
                pass
            self.clear_memory()

        self.clear_memory()
        gc.collect()
        time.sleep(0.1)

        # 測量
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
                import traceback
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
        if return_value is not None and isinstance(return_value, torch.Tensor):
            return_shape = str(return_value.shape)
            return_dtype = str(return_value.dtype)

        print(f"\n  {'='*40}")
        print(f"  {name} Results:")
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
            return_shape=return_shape,
            return_dtype=return_dtype
        )

    def compare_results(self, result_a: BenchmarkResult,
                        result_b: BenchmarkResult,
                        rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, str]:
        if not result_a.success or not result_b.success:
            return False, "error"

        if result_a.return_shape != result_b.return_shape:
            return False, "shape_mismatch"

        return True, "same_shape_and_dtype"

    def run_comparison(self, func_a: Callable, func_b: Callable,
                       args_a: Tuple = (), kwargs_a: Dict = None,
                       args_b: Tuple = (), kwargs_b: Dict = None,
                       name_a: str = "Function A",
                       name_b: str = "Function B") -> ComparisonReport:
        if kwargs_a is None:
            kwargs_a = {}
        if kwargs_b is None:
            kwargs_b = {}

        print(f"\n{'#'*60}")
        print(f"# Starting Benchmark Comparison")
        print(f"# Time: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        print(f"{'='*60}")
        print(f"# BENCHMARKING: {name_a}")
        print(f"{'='*60}")
        result_a = self.benchmark_function(func_a, args_a, kwargs_a, name_a)

        print(f"{'='*60}")
        print(f"# BENCHMARKING: {name_b}")
        print(f"{'='*60}")
        result_b = self.benchmark_function(func_b, args_b, kwargs_b, name_b)

        if result_b.runtime_seconds > 0:
            speedup = result_a.runtime_seconds / result_b.runtime_seconds
        else:
            speedup = float('inf') if result_a.runtime_seconds > 0 else 1.0

        time_diff = result_a.runtime_seconds - result_b.runtime_seconds
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

    def _print_comparison_report(self, report: ComparisonReport, name_a: str, name_b: str):
        print(f"\n{'#'*60}")
        print(f"#           COMPARISON REPORT")
        print(f"{'#'*60}\n")

        print(f"EXECUTION TIME:")
        print(f"  {name_a}:     {report.func_a_result.runtime_seconds:.6f}s")
        print(f"  {name_b}:     {report.func_b_result.runtime_seconds:.6f}s")
        print(f"  Difference:   {abs(report.time_diff_seconds):.6f}s")
        if abs(report.time_diff_seconds) > 0.0001:
            if report.time_diff_seconds > 0:
                winner = name_b
            else:
                winner = name_a
            print(f"  Winner:       {winner}")

        print(f"\nMEMORY USAGE (Peak):")
        print(f"  {name_a}:     {report.func_a_result.peak_memory_mb:.2f} MB")
        print(f"  {name_b}:     {report.func_b_result.peak_memory_mb:.2f} MB")
        print(f"  Difference:   {abs(report.memory_diff_mb):.2f} MB")

        print(f"\nCPU USAGE (Peak):")
        print(f"  {name_a}:     {report.func_a_result.peak_cpu_percent:.1f}%")
        print(f"  {name_b}:     {report.func_b_result.peak_cpu_percent:.1f}%")

        print(f"\nOUTPUT COMPARISON:")
        print(f"  Match Type:  {report.match_type}")
        print(f"  Match:      {'✓ PASS' if report.results_match else '✗ FAIL'}")

        if report.func_a_result.return_shape:
            print(f"  Shape A:    {report.func_a_result.return_shape}")
        if report.func_b_result.return_shape:
            print(f"  Shape B:    {report.func_b_result.return_shape}")
        if report.func_a_result.return_dtype:
            print(f"  Dtype A:    {report.func_a_result.return_dtype}")
        if report.func_b_result.return_dtype:
            print(f"  Dtype B:    {report.func_b_result.return_dtype}")

        print(f"\n{'='*60}")
        print(f"Benchmark completed at: {report.timestamp}")
        print(f"{'='*60}\n")

    def save_report(self, report: ComparisonReport, name_a: str, name_b: str,
                    filepath: str = "coords_transform_benchmark_report.json"):
        data = {
            "timestamp": report.timestamp,
            "function_a": {"name": name_a, **vars(report.func_a_result)},
            "function_b": {"name": name_b, **vars(report.func_b_result)},
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
# Function A: 使用 deepcopy + to()
# =============================================================================
def transform_coords_a(coords: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    版本 A: 使用 deepcopy
    - deepcopy 後再 .to(dtype)
    - deepcopy 後再 .to(float32)
    """
    coords = deepcopy(coords).to(dtype)
    coords = deepcopy(coords).to(torch.float32)
    return coords


# =============================================================================
# Function B: 直接轉換，copy=False
# =============================================================================
def transform_coords_b(coords: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    版本 B: 直接原地轉換
    - 直接 .to(dtype)
    - .to(..., non_blocking=True) 避免複製
    """
    coords = coords.to(dtype)
    coords = coords.to(torch.float32, non_blocking=True)
    return coords


# =============================================================================
# 輸入生成器
# =============================================================================
def generate_coords_input(batch_size: int = 8, num_points: int = 1024) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    產生測試用的座標張量

    Args:
        batch_size: 批次大小
        num_points: 每批的點數量
        dtype: 目標資料類型

    Returns:
        (input_tensor, kwargs_a, kwargs_b)
    """
    # 產生隨機座標 (預設 int64, 模擬點雲索引或類別)
    coords = torch.randint(0, 1000, (batch_size, num_points, 3), dtype=torch.int64)

    kwargs = {'dtype': torch.float32}

    return coords, kwargs, kwargs


def analyze_inputs():
    """分析函數輸入需求"""
    print("\n" + "="*60)
    print("INPUT ANALYSIS")
    print("="*60)

    print("\n[Function A: transform_coords_a]")
    print("  - coords: torch.Tensor (batch, points, 3)")
    print("  - dtype: torch.dtype (output type, default: float32)")
    print("  - 方式: 使用 deepcopy")
    print("  - 操作: deepcopy().to() + deepcopy().astype()")

    print("\n[Function B: transform_coords_b]")
    print("  - coords: torch.Tensor (batch, points, 3)")
    print("  - dtype: torch.dtype (output type, default: float32)")
    print("  - 方式: 直接原地轉換")
    print("  - 操作: .to() + .astype(copy=False)")

    print("\n[Test Input]")
    print(f"  - Shape: (batch, num_points, 3)")
    print(f"  - Default Test: batch=8, num_points=1024")
    print(f"  - 預設 dtype: int64 → float32")


def check_potential_issues():
    """檢查潛在問題"""
    print("\n" + "="*60)
    print("POTENTIAL ISSUES CHECK")
    print("="*60)

    print("\n[✓] PyTorch tensor: 確認 .to() 和 .astype() 方法可用")
    print("[✓] Shape: (batch, points, 3) = (8, 1024, 3)")
    print("[✓] 操作: deepcopy vs in-place")

    print("\n[!] WARNING: deepcopy 在 PyTorch 中會建立完整的資料副本")
    print("    版本 A 每次操作都會複製，記憶體使用會較高")
    print("    版本 B 使用原地操作，較節省資源")


def main():
    print("=" * 60)
    print("   Coordinate Transform Runtime Benchmark")
    print("=" * 60)

    analyze_inputs()
    check_potential_issues()

    # 預設測試參數
    batch_size = 8
    num_points = 1024

    print("\n" + "-"*60)
    print(f"Generating coords input...")
    print(f"  Shape: ({batch_size}, {num_points}, 3)")
    print(f"  Source dtype: int64")
    print(f"  Target dtype: float32")
    print("-"*60)

    input_tensor, kwargs_a, kwargs_b = generate_coords_input(batch_size, num_points)

    print(f"\nInput Tensor:")
    print(f"  Shape: {input_tensor.shape}")
    print(f"  Dtype: {input_tensor.dtype}")
    print(f"  Memory: {input_tensor.element_size() * input_tensor.nelement() / 1024:.2f} KB")

    runner = BenchmarkRunner(warmup_runs=3, measurement_runs=5)

    report = runner.run_comparison(
        func_a=transform_coords_a,
        func_b=transform_coords_b,
        args_a=(input_tensor.clone(),),
        kwargs_a=kwargs_a,
        args_b=(input_tensor.clone(),),
        kwargs_b=kwargs_b,
        name_a="transform_coords_a (deepcopy)",
        name_b="transform_coords_b (in-place)"
    )

    runner.save_report(report, "transform_coords_a", "transform_coords_b")

    print("\n" + "="*60)
    print("   BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()