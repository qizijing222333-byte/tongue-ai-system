import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _gm(vals: List[float]) -> float:
    vals = [max(1e-8, float(v)) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

def _get_prob(probs: Dict[str, Any], key: str) -> float:
    """
    Support BOTH:
    1) flat dict: {"body_color.pale": 0.78, ...}
    2) nested dict: {"body_color": {"pale": 0.78}, ...} (fallback)
    """
    # --- 1) flat dict fast path ---
    if isinstance(probs, dict) and key in probs:
        try:
            return float(probs[key])
        except Exception:
            return 0.0

    # --- 2) nested dict fallback ---
    parts = key.split(".")
    cur = probs
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return 0.0
    try:
        return float(cur)
    except Exception:
        return 0.0


def _match_condition(pred: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    # pred example:
    # {
    #   "body_color": {"label":"red","prob":0.8, "pale":0.1,"red":0.8,"dark":0.1},
    #   ...
    # }
    def eval_atom(atom: Dict[str, Any]) -> bool:
        t = atom["type"]
        if t == "eq":
            field = atom["field"]
            val = atom["value"]
            return pred.get(field, {}).get("label") == val
        if t == "any_label":
            field = atom["field"]
            values = set(atom["values"])
            return pred.get(field, {}).get("label") in values
        if t == "ge":
            field = atom["field"]  # e.g. features.tooth_mark
            v = atom["value"]
            # stored as pred["features"]["tooth_mark"]["prob"]
            if field.startswith("features."):
                feat = field.split(".", 1)[1]
                return float(pred.get("features", {}).get(feat, {}).get("prob", 0.0)) >= float(v)
            return False
        if t == "any_of":
            return any(eval_atom(x) for x in atom["items"])
        return False

    # support all_of only
    return all(eval_atom(x) for x in cond.get("all_of", []))

def _score_rule(pred: Dict[str, Any], score_cfg: Dict[str, Any]) -> float:
    if score_cfg["type"] != "gm":
        return 0.0
    terms = []
    for term in score_cfg["terms"]:
        if "field_prob" in term:
            terms.append(_get_prob(pred["probs_flat"], term["field_prob"]))
        elif "field_prob_max" in term:
            terms.append(max(_get_prob(pred["probs_flat"], k) for k in term["field_prob_max"]))
    return float(score_cfg.get("multiplier", 1.0)) * _gm(terms)

def _flatten_probs(pred: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    # for each task, we expect pred[task] has per-class probs as keys:
    # pred["body_color"]["red"]=0.8 etc.
    for task in ["body_color", "coat_color", "thickness", "greasy"]:
        for k, v in pred.get(task, {}).items():
            if k in ("label", "prob"):
                continue
            out[f"{task}.{k}"] = float(v)
    # features: pred["features"]["tooth_mark"]["prob"]
    for feat in ["tooth_mark", "fissure"]:
        out[f"features.{feat}"] = float(pred.get("features", {}).get(feat, {}).get("prob", 0.0))
    return out

class RuleEngine:
    def __init__(self, rules_path: str):
        self.rules_path = Path(rules_path)
        self.rules = json.loads(self.rules_path.read_text(encoding="utf-8"))

    def infer(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        # Attach flattened probs
        pred = dict(pred)
        pred["probs_flat"] = _flatten_probs(pred)

        policy = self.rules.get("decision_policy", {})
        topk = int(policy.get("topk", 2))
        low_th = float(policy.get("score_low_threshold", 55))
        delta = float(policy.get("mixed_delta", 10))

        results = []
        for c in self.rules["constitutions"]:
            best_rule = None
            best_score = 0.0
            for rule in sorted(c["rules"], key=lambda r: r.get("priority", 999)):
                if _match_condition(pred, rule["condition"]):
                    s = _score_rule(pred, rule["score"])
                    if s > best_score:
                        best_score = s
                        best_rule = rule
            # score scaled to 0-100
            score100 = best_score * 100.0
            conf = min(1.0, best_score / 0.85) if best_score > 0 else 0.0
            results.append({
                "name": c["name"],
                "display_name": c.get("display_name", c["name"]),
                "score": round(score100, 2),
                "confidence": round(conf, 3),
                "evidence": best_rule["evidence_text"] if best_rule else "证据不足，未命中明确规则。",
                "advice": best_rule["advice"] if best_rule else {
                    "diet": "建议在标准光照下重新采集舌象，并结合症状综合判断。",
                    "lifestyle": "规律作息，避免熬夜。",
                    "exercise": "适度运动。",
                    "medical": "如有不适或症状持续，请咨询专业医师。"
                }
            })

        # sort, take topk
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top = results_sorted[:topk]
        s1 = top[0]["score"] if top else 0
        s2 = top[1]["score"] if len(top) > 1 else 0

        if s1 < low_th:
            decision = "LOW_CONF"
        elif (s1 - s2) < delta:
            decision = "MIXED"
        else:
            decision = "CLEAR"

        return {
            "decision": decision,
            "topk": top,
            "all_scores": results_sorted
        }
