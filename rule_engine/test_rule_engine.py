from rule_engine.rule_engine import RuleEngine

# 伪造一个“模型输出概率”输入（用于测试规则引擎）
pred = {
    "body_color": {"label": "pale", "prob": 0.78, "pale": 0.78, "red": 0.15, "dark": 0.07},
    "coat_color": {"label": "white", "prob": 0.80, "white": 0.80, "yellow": 0.12, "no_coat": 0.08},
    "thickness": {"label": "thick", "prob": 0.70, "thin": 0.30, "thick": 0.70},
    "greasy": {"label": "greasy", "prob": 0.66, "non_greasy": 0.34, "greasy": 0.66},
    "features": {
        "tooth_mark": {"prob": 0.82},
        "fissure": {"prob": 0.10}
    }
}

engine = RuleEngine("rule_engine/rules_v2.json")
out = engine.infer(pred)

print("=== DECISION ===")
print(out["decision"])
print("=== TOPK ===")
for item in out["topk"]:
    print(item["name"], item["score"], item["confidence"])
print("=== FULL OUTPUT ===")
print(out)
