#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scorer.py — 多规则加权评分工具（纯标准库，单文件）

输入（JSON 文件或 stdin）：
{
  "objects": [ {"name": "obj1", "attributes": {"k": v, ...}}, ... ],
  "rules":   [ {"id": "r1",
                "when":      {"attr": "...", "op": "...", "value": ...} | null,
                "condition": {"attr": "...", "op": "...", "value": ...},
                "score": 10, "weight": 2}, ... ]
}

规则语义：when（适用条件）为真时规则适用于该对象；适用且 condition（属性条件）
为真时计分 score，权重 weight 参与汇总。

归一化规则（自定）：加权平均 normalized = Σ(w_i * s_i) / Σ(w_i)，
仅统计该对象实际适用并命中的规则。理由：
  1. 保尺度——结果与单条规则分值同量纲，可直接解释；
  2. 抗规则数量偏差——不同对象命中规则数不同，简单求和会惩罚命中少的对象；
  3. 权重即影响力——权重线性决定各规则对总分的贡献份额。

错误报告类型：
  invalid_rule            分值或权重非法（负数/非数值），该规则被剔除
  rule_conflict           两条规则对同一属性给出互斥（相反）条件的计分
  missing_attribute       规则引用了对象上不存在的属性
  illegal_attribute_value 属性值类型非法（如用数值比较作用于字符串）
  invalid_object          对象缺少 name 或 attributes 非字典

跨对象状态延续：--state FILE 指定状态文件，每次运行追加一条带 run_id、
时间戳、逐规则贡献明细的记录，评分历史可追溯。

用法：
  python3 scorer.py input.json [--state state.json] [--pretty]
  cat input.json | python3 scorer.py - [--pretty]
  python3 scorer.py --selftest      # 运行内置自测样例
"""

import argparse
import json
import sys
import time
import uuid


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _op_eq(a, b):   return a == b
def _op_ne(a, b):   return a != b
def _op_lt(a, b):   return a < b
def _op_le(a, b):   return a <= b
def _op_gt(a, b):   return a > b
def _op_ge(a, b):   return a >= b
def _op_in(a, b):   return a in b
def _op_contains(a, b): return b in a

OPS = {
    "==": _op_eq, "!=": _op_ne, "<": _op_lt, "<=": _op_le,
    ">": _op_gt, ">=": _op_ge, "in": _op_in, "contains": _op_contains,
}
NUMERIC_OPS = {"<", "<=", ">", ">="}


# ---------------------------------------------------------------- 校验

def validate_rules(raw_rules, errors):
    """剔除非法规则（分值/权重为负或非数值、缺字段、未知操作符）。"""
    valid = []
    for idx, rule in enumerate(raw_rules):
        rid = rule.get("id", f"<rule#{idx}>")
        cond = rule.get("condition")
        if not isinstance(cond, dict) or "attr" not in cond or "op" not in cond:
            errors.append({"type": "invalid_rule", "rule": rid,
                           "message": "规则缺少合法的 condition（需含 attr/op）"})
            continue
        for phase in ("when", "condition"):
            c = rule.get(phase)
            if c is None:
                continue
            if c.get("op") not in OPS and not (phase == "when" and c.get("op") == "exists"):
                errors.append({"type": "invalid_rule", "rule": rid,
                               "message": f"{phase} 使用未知操作符: {c.get('op')!r}"})
        score, weight = rule.get("score"), rule.get("weight")
        bad = False
        if not _is_num(score) or score < 0:
            errors.append({"type": "invalid_rule", "rule": rid,
                           "message": f"分值非法（须为非负数值）: {score!r}"})
            bad = True
        if not _is_num(weight) or weight < 0:
            errors.append({"type": "invalid_rule", "rule": rid,
                           "message": f"权重非法（须为非负数值）: {weight!r}"})
            bad = True
        if bad:
            continue
        valid.append(rule)
    return valid


def _contradictory(c1, c2):
    """判断同一属性上的两个条件是否互斥（覆盖常见的相等/不等/区间组合）。"""
    o1, o2 = c1.get("op"), c2.get("op")
    v1, v2 = c1.get("value"), c2.get("value")
    pair = {o1, o2}
    if pair == {"=="}:
        return v1 != v2
    if pair == {"==", "!="}:
        return v1 == v2
    if pair == {">=", "<"} or pair == {">", "<="}:
        return v1 >= v2 if o1 in (">=", ">") else v2 >= v1
    if pair == {"<=", ">"} or pair == {"<", ">="}:
        return v1 <= v2 if o1 in ("<=", "<") else v2 <= v1
    return False


def detect_conflicts(rules, errors):
    """同一属性上条件互斥却各自计分的规则对 → 冲突报告。"""
    by_attr = {}
    for r in rules:
        by_attr.setdefault(r["condition"]["attr"], []).append(r)
    reported = set()
    for attr, rs in by_attr.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if _contradictory(rs[i]["condition"], rs[j]["condition"]):
                    key = tuple(sorted((rs[i]["id"], rs[j]["id"])))
                    if key not in reported:
                        reported.add(key)
                        errors.append({
                            "type": "rule_conflict", "attribute": attr,
                            "rules": sorted(key),
                            "message": f"规则 {key[0]} 与 {key[1]} 对属性 "
                                       f"'{attr}' 给出互斥条件的计分"})


def validate_object(obj, errors):
    name = obj.get("name")
    attrs = obj.get("attributes")
    if not name or not isinstance(attrs, dict):
        errors.append({"type": "invalid_object", "object": name,
                       "message": "对象缺少 name 或 attributes 非字典"})
        return None, None
    for k, v in attrs.items():
        if v is None or isinstance(v, (dict,)) or (
                isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)):
            errors.append({"type": "illegal_attribute_value", "object": name,
                           "attribute": k,
                           "message": f"属性值类型非法: {v!r}"})
    return name, attrs


# ---------------------------------------------------------------- 求值

def eval_condition(cond, attrs, errors, obj_name, rule_id, phase):
    """返回 True/False；出错时记录错误并返回 None（视为不命中）。"""
    attr, op = cond.get("attr"), cond.get("op")
    if op == "exists":
        return attr in attrs
    if attr not in attrs:
        errors.append({"type": "missing_attribute", "object": obj_name,
                       "attribute": attr, "rule": rule_id,
                       "message": f"规则 {rule_id} 的 {phase} 引用了对象 "
                                  f"'{obj_name}' 上不存在的属性 '{attr}'"})
        return None
    a, b = attrs[attr], cond.get("value")
    if op in NUMERIC_OPS and not (_is_num(a) and _is_num(b)):
        errors.append({"type": "illegal_attribute_value", "object": obj_name,
                       "attribute": attr, "rule": rule_id,
                       "message": f"数值比较 {op} 作用于非数值: {a!r} vs {b!r}"})
        return None
    if op == "in" and not isinstance(b, (list, tuple, str)):
        errors.append({"type": "invalid_rule", "rule": rule_id,
                       "message": f"in 操作的目标须为列表/字符串: {b!r}"})
        return None
    if op == "contains" and not isinstance(a, (list, tuple, str)):
        errors.append({"type": "illegal_attribute_value", "object": obj_name,
                       "attribute": attr, "rule": rule_id,
                       "message": f"contains 要求属性值为列表/字符串: {a!r}"})
        return None
    try:
        return bool(OPS[op](a, b))
    except TypeError as exc:
        errors.append({"type": "illegal_attribute_value", "object": obj_name,
                       "attribute": attr, "rule": rule_id,
                       "message": f"比较失败: {exc}"})
        return None


def score_object(name, attrs, rules, errors):
    """对单个对象执行全部规则，返回含逐规则明细的结果。"""
    breakdown, raw, wsum = [], 0.0, 0.0
    for rule in rules:
        rid = rule["id"]
        when = rule.get("when")
        if when is not None:
            applicable = eval_condition(when, attrs, errors, name, rid, "when")
            if applicable is not True:
                continue
        hit = eval_condition(rule["condition"], attrs, errors, name, rid, "condition")
        if hit is True:
            w, s = rule["weight"], rule["score"]
            breakdown.append({"rule": rid, "weight": w, "score": s,
                              "contribution": w * s})
            raw += w * s
            wsum += w
    normalized = raw / wsum if wsum > 0 else 0.0
    return {"object": name, "raw_score": raw, "weight_sum": wsum,
            "normalized": round(normalized, 6), "breakdown": breakdown}


# ---------------------------------------------------------------- 主流程

def run(payload, state_path=None):
    errors, results = [], []
    rules = validate_rules(payload.get("rules", []), errors)
    detect_conflicts(rules, errors)
    for obj in payload.get("objects", []):
        name, attrs = validate_object(obj, errors)
        if name is None:
            continue
        results.append(score_object(name, attrs, rules, errors))

    report = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
        "errors": errors,
        "summary": {"objects": len(results), "rules_used": len(rules),
                    "error_count": len(errors)},
    }

    if state_path:
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = {"runs": []}
        state.setdefault("runs", []).append(
            {"run_id": report["run_id"], "timestamp": report["timestamp"],
             "results": results, "errors": errors})
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        report["state_file"] = state_path
        report["history_runs"] = len(state["runs"])
    return report


# ---------------------------------------------------------------- 自测

SELFTEST_INPUT = {
    "objects": [
        {"name": "alpha", "attributes": {"level": 5, "region": "cn",
                                         "tags": ["vip"], "active": True}},
        {"name": "beta",  "attributes": {"level": 3, "region": "us",
                                         "active": False}},
        {"name": "gamma", "attributes": {"level": "high", "region": "cn"}},
    ],
    "rules": [
        {"id": "r1", "when": {"attr": "level", "op": ">=", "value": 3},
         "condition": {"attr": "region", "op": "==", "value": "cn"},
         "score": 10, "weight": 2},
        {"id": "r2", "when": None,
         "condition": {"attr": "active", "op": "==", "value": True},
         "score": 5, "weight": 1},
        {"id": "r3", "when": None,
         "condition": {"attr": "region", "op": "==", "value": "cn"},
         "score": -4, "weight": 1},
        {"id": "r4", "when": None,
         "condition": {"attr": "level", "op": ">=", "value": 4},
         "score": 8, "weight": 1},
        {"id": "r5", "when": None,
         "condition": {"attr": "level", "op": "<", "value": 4},
         "score": 6, "weight": 1},
        {"id": "r6", "when": None,
         "condition": {"attr": "tags", "op": "contains", "value": "vip"},
         "score": 3, "weight": 1},
    ],
}


def selftest():
    import os, tempfile
    fd, state_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(state_file)
    try:
        report = run(SELFTEST_INPUT, state_path=state_file)
        run(SELFTEST_INPUT, state_path=state_file)  # 第二次运行验证状态延续

        res = {r["object"]: r for r in report["results"]}
        etypes = {e["type"] for e in report["errors"]}

        # alpha: r1(2*10)+r2(1*5)+r4(1*8)+r6(1*3)=36, wsum=5 → 7.2
        assert res["alpha"]["normalized"] == 7.2, res["alpha"]
        # beta: 仅 r5(1*6) 命中 → 6.0
        assert res["beta"]["normalized"] == 6.0, res["beta"]
        # gamma: level 为字符串，全部数值规则报错，无命中 → 0.0
        assert res["gamma"]["normalized"] == 0.0, res["gamma"]
        # 五类错误齐备
        assert {"invalid_rule", "rule_conflict", "missing_attribute",
                "illegal_attribute_value"} <= etypes, etypes
        # r3 负分值被剔除
        assert any(e["type"] == "invalid_rule" and e.get("rule") == "r3"
                   for e in report["errors"])
        # r4/r5 在 level 上互斥 → 冲突
        assert any(e["type"] == "rule_conflict" and e["rules"] == ["r4", "r5"]
                   for e in report["errors"])
        # 状态延续：两次运行都入档
        with open(state_file, encoding="utf-8") as f:
            assert len(json.load(f)["runs"]) == 2

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("\n[selftest] 全部断言通过 ✔", file=sys.stderr)
    finally:
        if os.path.exists(state_file):
            os.unlink(state_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description="多规则加权评分工具")
    ap.add_argument("input", nargs="?", help="输入 JSON 文件路径，'-' 表示 stdin")
    ap.add_argument("--state", help="状态文件路径（跨运行延续评分历史）")
    ap.add_argument("--pretty", action="store_true", help="缩进美化输出")
    ap.add_argument("--selftest", action="store_true", help="运行内置自测样例")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if not args.input:
        ap.error("需要输入文件（或 --selftest）")
    try:
        text = sys.stdin.read() if args.input == "-" \
            else open(args.input, encoding="utf-8").read()
        payload = json.loads(text)
    except (OSError, ValueError) as exc:
        print(f"输入读取/解析失败: {exc}", file=sys.stderr)
        return 2
    report = run(payload, state_path=args.state)
    kw = {"ensure_ascii": False}
    if args.pretty:
        kw["indent"] = 2
    print(json.dumps(report, **kw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
