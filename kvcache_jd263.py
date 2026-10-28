"""kvcache_jd263 —— 分析KV Cache 与显存优化工程师·成本与吞吐的帕累托边界的数值分布并标出离群点。

抽取数值做分布分析，用中位数绝对偏差识别KV Cache 与显存优化工程师·成本与吞吐的帕累托边界上的离群点

定位：
  用中位数绝对偏差而不是标准差，离群点本身不会把阈值带偏

岗位对应：
  目标岗位　无问芯穹 · 架构KV Cache 与显存优化工程师
  技能主线　推理加速、CUDA、大模型推理
  领域标签　大模型推理、AI基础设施

设计约束：
  1. **只报告能定位到具体位置的结论**，不做语义猜测。
  2. **规则表公开**，每条命中都能对应到岗位要求的一条明示规则。
  3. **输入为空或格式不对时明确报错**，不静默返回空结果。

仅依赖 Python 标准库。

用法:
    python kvcache_jd263.py --help
    python kvcache_jd263.py --file input.txt
    python kvcache_jd263.py --text "示例内容"
    python kvcache_jd263.py --file input.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# 岗位规则表：每条规则独立可核查，命中即报，不做推断。
# 规则由岗位 JD 的技能线与领域标签派生——每条都能对回岗位要求。
# 目标岗位：无问芯穹 · 架构KV Cache 与显存优化工程师
RULES: list[dict] = [
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·缺失",
        "why": "岗位要求「推理加速」在输入里找不到对应内容",
        "severity": "high",
        "keywords": ['missing', 'absent', '缺少', '没有', '未提供', '推理加速'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·冲突",
        "why": "两处关于「CUDA」的描述互相矛盾",
        "severity": "high",
        "keywords": ['conflict', 'contradict', '矛盾', '不一致', '相反', 'CUDA'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·越界",
        "why": "「大模型推理」的取值超出岗位场景的常规范围",
        "severity": "medium",
        "keywords": ['out of range', 'outlier', '越界', '超限', '超过', '大模型推理'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·退化",
        "why": "「AI基础设施」相关的信息量不足以支撑判断",
        "severity": "medium",
        "keywords": ['degrade', 'insufficient', '不足', '太少', '无法', 'AI基础设施'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·泄漏",
        "why": "「KV Cache 与显存优化工程师」相关的内容被意外暴露",
        "severity": "high",
        "keywords": ['leak', 'expose', '泄漏', '暴露', '明文', 'KV Cache 与显存优化工程师'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·重复#2",
        "why": "同一处「推理加速」信息被反复传递",
        "severity": "low",
        "keywords": ['duplicate', 'repeat', '重复', 'again', '推理加速'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·陈旧#2",
        "why": "「CUDA」的时效性已过期",
        "severity": "medium",
        "keywords": ['stale', 'expired', '过期', '陈旧', '旧版', 'CUDA'],
    },
    {
        "name": "KV Cache 与显存优化工程师·成本与吞吐的帕累托边界·不可核#2",
        "why": "关于「大模型推理」的说法无法追溯到具体证据",
        "severity": "high",
        "keywords": ['unverifiable', 'no evidence', '无法追溯', '没有来源', '大模型推理'],
    },
]


def kvcache_jd263_read_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def kvcache_jd263_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    pos = (len(vs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vs) - 1)
    frac = pos - lo
    return vs[lo] * (1 - frac) + vs[hi] * frac


def kvcache_jd263_mad(values: list[float]) -> float:
    """中位数绝对偏差——比标准差更抗离群点。"""
    if not values:
        return 0.0
    med = kvcache_jd263_percentile(values, 0.5)
    return kvcache_jd263_percentile([abs(v - med) for v in values], 0.5)


def kvcache_jd263_detect_outliers(values: list[float], k: float = 3.0) -> list[dict]:
    """用中位数与 MAD 找离群点。"""
    if not values:
        return []
    med = kvcache_jd263_percentile(values, 0.5)
    scale = kvcache_jd263_mad(values)
    if scale == 0:
        return []
    out = []
    for i, v in enumerate(values):
        score = abs(v - med) / (1.4826 * scale)
        if score > k:
            out.append({"index": i, "value": v, "robust_z": round(score, 3)})
    return out


def kvcache_jd263_analyze(text: str) -> dict:
    """主分析：分布形状、稳健统计与离群点。"""
    nums = kvcache_jd263_read_numbers(text)
    if not nums:
        return {"ok": False, "error": "文本里没有数字", "count": 0}
    n = len(nums)
    mean = sum(nums) / n
    var = sum((v - mean) ** 2 for v in nums) / n
    buckets: dict[str, int] = {}
    for v in nums:
        b = ("<0" if v < 0 else "0-10" if v < 10 else "10-100" if v < 100
             else "100-1000" if v < 1000 else ">=1000")
        buckets[b] = buckets.get(b, 0) + 1
    return {
        "ok": True,
        "count": n,
        "mean": mean,
        "stdev": var ** 0.5,
        "median": kvcache_jd263_percentile(nums, 0.5),
        "mad": kvcache_jd263_mad(nums),
        "min": min(nums),
        "max": max(nums),
        "outliers": kvcache_jd263_detect_outliers(nums),
        "buckets": buckets,
        "values": nums[:50],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    """把结果渲染成人读的文本。每个工具的版式不同。"""
    out: list[str] = []
    if not result.get("ok", True):
        out.append(f"无法分析：{{result.get('error', '未知原因')}}")
        return "\n".join(out)
    out.append(f"样本数：{result['count']}")
    out.append(f"均值 {result['mean']:.4f}  标准差 {result['stdev']:.4f}  中位 {result['median']:.4f}")
    out.append(f"MAD {result['mad']:.4f}  区间 [{result['min']}, {result['max']}]")
    out.append("")
    out.append("分桶：")
    for b, c in sorted(result["buckets"].items()):
        out.append(f"  {b:>10}  {c}")
    if result["outliers"]:
        out.append("")
        out.append("离群点：")
        for o in result["outliers"]:
            out.append(f"  #{o['index']} = {o['value']}  (robust z={o['robust_z']})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kvcache_jd263",
        description="抽取数值做分布分析，用中位数绝对偏差识别KV Cache 与显存优化工程师·成本与吞吐的帕累托边界上的离群点（仅依赖标准库）",
        epilog=(
            "示例:\n"
            "  python kvcache_jd263.py --file input.txt\n"
            "  python kvcache_jd263.py --text '示例内容'\n"
            "  python kvcache_jd263.py --file input.txt --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="输入文件路径")
    src.add_argument("--text", help="直接给出输入内容")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            print(f"读不了文件：{{e}}", file=sys.stderr)
            return 2
    else:
        content = args.text

    if not content.strip():
        print("输入是空的。", file=sys.stderr)
        return 2

    result = kvcache_jd263_analyze(content)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(render(result))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
