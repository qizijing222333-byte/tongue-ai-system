import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import cv2

from inference.quality_gate import load_config, quality_gate
from rule_engine.rule_engine import RuleEngine

import os

def _get_model_version(predictor) -> str:
    """
    FAKE/REAL version switch:
    - predictor is None or predictor.ready is False -> FAKE-0.0
    - predictor.ready is True -> REAL-<weights filename or custom tag>
    """
    if predictor is None:
        return "FAKE-0.0"

    # 如果 predictor 提供 ready 字段，优先用它
    if hasattr(predictor, "ready") and (not predictor.ready):
        return "FAKE-0.0"

    # 版本号优先用 predictor.model_version（你之后可在 predictor 里写死 REAL-v1.0）
    mv = getattr(predictor, "model_version", "") or ""
    if mv:
        return mv

    # 否则用权重文件名
    wp = getattr(predictor, "weights_path", "") or ""
    name = os.path.basename(wp) if wp else "unknown"
    return f"REAL-{name}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fake_predictor() -> Dict[str, Any]:
    """
    Placeholder for model outputs.
    Later you will replace this with:
      - detector -> bbox
      - segmenter -> mask + area ratio
      - multitask classifier -> probabilities
      - feature heads -> tooth_mark/fissure probs
    """
    return {
        "body_color": {"label": "pale", "prob": 0.78, "pale": 0.78, "red": 0.15, "dark": 0.07},
        "coat_color": {"label": "white", "prob": 0.80, "white": 0.80, "yellow": 0.12, "no_coat": 0.08},
        "thickness": {"label": "thick", "prob": 0.70, "thin": 0.30, "thick": 0.70},
        "greasy": {"label": "greasy", "prob": 0.66, "non_greasy": 0.34, "greasy": 0.66},
        "features": {
            "tooth_mark": {"prob": 0.82},
            "fissure": {"prob": 0.10}
        }
    }


def _format_detect(found: bool, conf: float = 1.0, bbox: Optional[list] = None) -> Dict[str, Any]:
    return {"found": found, "conf": float(conf), "bbox": bbox}


def _format_segment(conf: float = 1.0, tongue_area_ratio: Optional[float] = None) -> Dict[str, Any]:
    return {"conf": float(conf), "tongue_area_ratio": tongue_area_ratio}


def run_pipeline(image_bgr, cfg: Dict[str, Any], predictor=None, return_debug: bool = False) -> Dict[str, Any]:
    # ---- Meta ----
    out: Dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "model_version": _get_model_version(predictor),
        "rule_version": "v2.0",
    }

    # ---- Quality Gate ----
    q = quality_gate(image_bgr, cfg)
    out["quality"] = {
        "pass": q.passed,
        "reason_codes": q.reason_codes,
        "user_tip": q.user_tip,
    }

    # ---- Default artifacts ----
    out["artifacts"] = {
        "roi_image_url": None,
        "mask_image_url": None,
        "debug_images_url": None
    }

    if return_debug:
        out["quality"]["debug"] = q.debug

    # If not pass -> stop early
    if not q.passed:
        out["detect"] = None
        out["segment"] = None
        out["tongue"] = None
        out["constitution"] = None
        return out

    # ---- Detection / Segmentation placeholders ----
    # Later: replace with real detector/segmenter outputs
    out["detect"] = _format_detect(found=True, conf=1.0, bbox=None)
    out["segment"] = _format_segment(conf=1.0, tongue_area_ratio=q.debug.get("rough_tongue_area_ratio"))

     # ---- Model prediction ----
    if predictor is None:
        pred = _fake_predictor()
    else:
        pred = predictor.predict(image_bgr)

    out["tongue"] = pred

    # ---- Rule engine ----
    engine = RuleEngine(cfg["rules"]["file"])
    ce = engine.infer(pred)
    out["constitution"] = {
        "decision": ce["decision"],
        "topk": [
            {
                "name": x["name"],
                "score": x["score"],
                "confidence": x["confidence"],
                "evidence": x["evidence"],
                "advice": x["advice"]
            } for x in ce["topk"]
        ]
    }

    return out


if __name__ == "__main__":
    import sys
    cfg = load_config()

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    result = run_pipeline(img, cfg, return_debug=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))

