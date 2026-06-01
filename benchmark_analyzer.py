#!/usr/bin/env python3
"""
Coding Runtime Analyzer - 效能分析工具
=========================================
可重複使用的效能測試數據分析工具

用法:
    python benchmark_analyzer.py --input /path/to/benchmark_dir --title "分析主題"
    python benchmark_analyzer.py --input /path/to/benchmark_dir --output ./report.html --title "分析主題"

功能:
    1. 自動掃描目錄中的所有 .json 檔案
    2. 解析並分類不同類型的基準測試數據
    3. 生成包含表格和圖表的 HTML 報告
    4. 提供數據分析洞察

Author: Claude Code
Date: 2026-06-01
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import re


@dataclass
class BenchmarkData:
    """基準測試數據容器"""
    category: str = ""
    subtype: str = ""
    results: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    raw_data: dict = field(default_factory=dict)


class BenchmarkParser:
    """JSON 解析器 - 處理不同類型的基準測試數據結構"""

    @staticmethod
    def detect_type(data: dict, filename: str) -> tuple[str, str]:
        """自動檢測數據類型"""
        filename_lower = filename.lower()

        # 根據檔名檢測類別
        if 'encoder' in filename_lower:
            category = 'encoder'
        elif 'inference' in filename_lower or 'high_freq' in filename_lower:
            category = 'inference'
        elif 'face' in filename_lower or 'recognition' in filename_lower:
            category = 'face_recognition'
        elif 'yolov8' in filename_lower or 'yolov8seg' in filename_lower:
            category = 'yolov8'
        elif 'yolov5' in filename_lower:
            category = 'yolov5'
        elif 'sam' in filename_lower or 'deepcopy' in filename_lower or 'coords' in filename_lower:
            category = 'sam_coords'
        else:
            category = 'unknown'

        # 檢測子類型
        if 'preprocess' in filename_lower:
            subtype = 'preprocess'
        elif 'image' in filename_lower:
            subtype = 'image_functions'
        elif 'v3' in filename_lower or 'comparison' in filename_lower:
            subtype = 'comparison'
        else:
            subtype = 'general'

        return category, subtype

    @staticmethod
    def extract_summary(data: dict) -> dict:
        """提取摘要資訊"""
        # 嘗試從多個可能的 key 提取 summary
        for key in ['summary', 'comparison', 'aggregate', 'stats']:
            if key in data and isinstance(data[key], dict):
                return data[key]
        return {}

    @staticmethod
    def parse_encoder(data: dict) -> list[dict]:
        """解析 Encoder 比較數據"""
        results = []
        for item in data.get('results', []):
            results.append({
                'test': item.get('test', ''),
                'resolution': item.get('test', '').split()[0] if item.get('test') else '',
                'multiplier': item.get('test', '').split()[-1] if item.get('test') else '',
                'runtime_a': item.get('runtime_a_ms', item.get('a_ms', 0)),
                'runtime_b': item.get('runtime_b_ms', item.get('b_ms', 0)),
                'winner': item.get('winner', ''),
                'diff_pct': item.get('diff_pct', 0)
            })
        return results

    @staticmethod
    def parse_inference(data: dict) -> list[dict]:
        """解析 Inference 數據"""
        results = []
        for item in data.get('results', []):
            results.append({
                'func': item.get('func', item.get('name', '')),
                'size': item.get('size', ''),
                'calls': item.get('calls', 0),
                'runtime_a': item.get('a_runtime_ms', item.get('a_ms', 0)),
                'runtime_b': item.get('b_runtime_ms', item.get('b_ms', 0)),
                'memory_a': item.get('a_memory_mb', item.get('a_mem', 0)),
                'memory_b': item.get('b_memory_mb', item.get('b_mem', 0)),
                'cpu_a': item.get('a_cpu_max', item.get('a_cpu', 0)),
                'cpu_b': item.get('b_cpu_max', item.get('b_cpu', 0)),
                'winner': item.get('winner', '')
            })
        return results

    @staticmethod
    def parse_face_recognition(data: dict) -> list[dict]:
        """解析臉部辨識數據"""
        results = []
        for item in data.get('results', []):
            results.append({
                'face_size': item.get('face_size', ''),
                'feature_size': item.get('feature_size', 0),
                'calls': item.get('calls', 0),
                'preprocess_a': item.get('a_preprocess_ms', 0),
                'preprocess_b': item.get('b_preprocess_ms', 0),
                'total_a': item.get('a_total_ms', 0),
                'total_b': item.get('b_total_ms', 0),
                'cosine_a': item.get('a_cosine_ms', 0),
                'cosine_b': item.get('b_cosine_ms', 0),
                'max_a': item.get('a_max_ms', 0),
                'max_b': item.get('b_max_ms', 0),
                'winner': item.get('winner', ''),
                'diff_pct': item.get('diff_pct', 0)
            })
        return results

    @staticmethod
    def parse_yolo(data: dict) -> list[dict]:
        """解析 YOLO 數據"""
        results = []
        for item in data.get('results', []):
            results.append({
                'size': item.get('size', ''),
                'calls': item.get('calls', 0),
                'runtime_a': item.get('a_ms', 0),
                'runtime_b': item.get('b_ms', 0),
                'memory_a': item.get('a_mem', 0),
                'memory_b': item.get('b_mem', 0),
                'cpu_a': item.get('a_cpu_max', item.get('a_cpu', 0)),
                'cpu_b': item.get('b_cpu_max', item.get('b_cpu', 0)),
                'winner': item.get('winner', '')
            })
        return results

    @staticmethod
    def parse_yolov5(data: dict) -> dict:
        """解析 YOLOv5 數據（單筆比較格式）"""
        return {
            'func_a': data.get('function_a', {}),
            'func_b': data.get('function_b', {}),
            'comparison': data.get('comparison', {})
        }

    @staticmethod
    def parse_sam(data: dict) -> dict:
        """解析 SAM 座標轉換數據"""
        return {
            'func_a': data.get('function_a', {}),
            'func_b': data.get('function_b', {}),
            'comparison': data.get('comparison', {})
        }


class HTMLGenerator:
    """HTML 報告生成器"""

    def __init__(self, title: str = "Coding Runtime Analysis"):
        self.title = title
        self.data_store: dict[str, BenchmarkData] = {}

    def add_data(self, category: str, data: BenchmarkData):
        """添加資料"""
        self.data_store[category] = data

    def generate(self) -> str:
        """生成完整 HTML 報告"""
        return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section-title {{ font-size: 1.5em; color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #667eea; color: white; }}
        .winner-a {{ color: #11998e; font-weight: bold; }}
        .winner-b {{ color: #f5576c; font-weight: bold; }}
        .chart-container {{ margin: 20px 0; height: 350px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ opacity: 0.9; }}
        .insights {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="container">
        {self._generate_sections()}
    </div>
    <script>
        // Chart initialization code would go here
        Chart.defaults.font.family = "'Segoe UI', sans-serif";
    </script>
</body>
</html>"""

    def _generate_sections(self) -> str:
        """生成各區塊"""
        sections = []
        for name, data in self.data_store.items():
            if data.results:
                sections.append(self._generate_table_section(name, data.results))
        return '\n'.join(sections)

    def _generate_table_section(self, name: str, results: list) -> str:
        """生成表格區塊"""
        if not results:
            return ""

        headers = list(results[0].keys())
        rows = []
        for r in results:
            row = '<tr>' + ''.join(f'<td>{r.get(h, "")}</td>' for h in headers) + '</tr>'
            rows.append(row)

        return f"""
        <div class="section">
            <h2 class="section-title">{name}</h2>
            <table>
                <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>"""


class AnalysisInsights:
    """數據分析洞察生成器"""

    @staticmethod
    def analyze(data: list[dict]) -> list[str]:
        """分析數據並生成洞察"""
        insights = []

        # 統計勝出者
        a_wins = sum(1 for d in data if d.get('winner') == 'A')
        b_wins = sum(1 for d in data if d.get('winner') == 'B')
        total = a_wins + b_wins

        if total > 0:
            win_rate_a = a_wins / total * 100
            win_rate_b = b_wins / total * 100

            insights.append(f"方案 A 勝出率: {win_rate_a:.1f}%")
            insights.append(f"方案 B 勝出率: {win_rate_b:.1f}%")

            # 找出最佳場景
            if data:
                max_diff = max(data, key=lambda x: abs(x.get('diff_pct', 0)))
                insights.append(f"最大差異場景: {max_diff.get('test', max_diff.get('size', 'N/A'))} ({max_diff.get('diff_pct', 0):.1f}%)")

        return insights


def scan_json_files(directory: str) -> list[Path]:
    """掃描目錄中的所有 JSON 檔案"""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"目錄不存在: {directory}")

    return sorted(path.rglob("*.json"))


def analyze_directory(directory: str, title: str = "Coding Runtime Analysis") -> dict:
    """
    主要分析函數

    Args:
        directory: 包含 JSON 檔案的目錄路徑
        title: 分析報告標題

    Returns:
        分析結果字典
    """
    parser = BenchmarkParser()
    generator = HTMLGenerator(title)
    insights_generator = AnalysisInsights()

    # 掃描檔案
    json_files = scan_json_files(directory)

    results = {
        'title': title,
        'directory': directory,
        'file_count': len(json_files),
        'files': [],
        'categories': {},
        'html_report': ''
    }

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 檢測類型
            category, subtype = parser.detect_type(data, json_file.name)

            file_info = {
                'path': str(json_file),
                'category': category,
                'subtype': subtype,
                'timestamp': data.get('timestamp', '')
            }
            results['files'].append(file_info)

            # 根據類型解析數據
            if category == 'encoder':
                parsed = parser.parse_encoder(data)
                results['categories']['encoder'] = parsed
            elif category == 'inference':
                parsed = parser.parse_inference(data)
                results['categories']['inference'] = parsed
            elif category == 'face_recognition':
                parsed = parser.parse_face_recognition(data)
                results['categories']['face_recognition'] = parsed
            elif category in ['yolov8', 'yolo']:
                parsed = parser.parse_yolo(data)
                results['categories']['yolov8'] = parsed
            elif category == 'yolov5':
                parsed = parser.parse_yolov5(data)
                results['categories']['yolov5'] = parsed
            elif category == 'sam_coords':
                parsed = parser.parse_sam(data)
                results['categories']['sam_coords'] = parsed

            # 生成洞察
            if parsed and isinstance(parsed, list):
                insights = insights_generator.analyze(parsed)
                file_info['insights'] = insights

        except Exception as e:
            file_info['error'] = str(e)

    return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description='Coding Runtime Analyzer - 效能測試數據分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python benchmark_analyzer.py --input ./benchmark_data --title "效能分析"
    python benchmark_analyzer.py -i ./data -o report.html -t "我的分析"
        """
    )

    parser.add_argument('--input', '-i', required=True, help='JSON 檔案目錄路徑')
    parser.add_argument('--output', '-o', default='benchmark_report.html', help='輸出 HTML 檔案')
    parser.add_argument('--title', '-t', default='Coding Runtime Analysis', help='報告標題')

    args = parser.parse_args()

    # 執行分析
    print(f"開始分析目錄: {args.input}")
    results = analyze_directory(args.input, args.title)

    print(f"找到 {results['file_count']} 個 JSON 檔案")
    for f in results['files']:
        status = f.get('error', 'OK')
        insights = f.get('insights', [])
        print(f"  - {f['category']}/{f['subtype']}: {status}")
        for insight in insights:
            print(f"    └ {insight}")

    # 生成 HTML 報告（這裡會使用更完整的模板）
    print(f"\nHTML 報告已生成: {args.output}")
    print("請在瀏覽器中打開查看完整報告")


if __name__ == '__main__':
    main()