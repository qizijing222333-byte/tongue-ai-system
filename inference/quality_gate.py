from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import yaml

DEFAULT_REASON_TIPS = {
    "BLUR": "图片不清晰：请保持相机稳定并对准舌面重新拍摄。",
    "OVEREXPOSED": "图片过曝：请降低光线强度，避免反光或直射灯光。",
    "UNDEREXPOSED": "图片欠曝：请增加光线或靠近补光源重新拍摄。",
    "COLOR_CAST": "图片偏色：请在白光环境拍摄，避免偏黄/彩灯光源。",
    "LOW_RESOLUTION": "图片分辨率过低：请更近拍摄或使用更清晰的设备。",
    "NO_TONGUE_SUSPECTED": "未检测到明显舌体：请伸舌居中、对焦后重新拍摄。",
    "TONGUE_TOO_SMALL": "舌体区域太小：请靠近一些拍摄，确保舌头占画面较大比例。",
    "OCCLUSION_OR_BACKGROUND": "舌体可能被遮挡或背景干扰：请保持口部清晰、背景简单并重新拍摄。"
}

@dataclass
class QualityResult:
    passed: bool
    reason_codes: List[str]
    user_tip: str
    debug: Dict[str, Any]

def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _exposure_ratios(gray: np.ndarray) -> Tuple[float, float]:
    over = float((gray >= 245).mean())
    under = float((gray <= 10).mean())
    return over, under

def _color_cast_score(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr)
    mb, mg, mr = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
    m = (mb + mg + mr) / 3.0
    dev = (abs(mb - m) + abs(mg - m) + abs(mr - m)) / (3.0 * (m + 1e-6))
    return float(dev)

def _rough_tongue_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic tongue-like region mask using HSV.
    This is a v2 placeholder before detector/segmenter is available.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Tongue-like colors: red/pink/flesh. HSV has red wrap-around.
    lower1 = np.array([0, 35, 40], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 35, 40], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def quality_gate(image_bgr: np.ndarray, cfg: Dict[str, Any]) -> QualityResult:
    th = cfg["quality_gate"]["thresholds"]
    reasons: List[str] = []
    debug: Dict[str, Any] = {}

    h, w = image_bgr.shape[:2]
    debug["shape"] = [h, w]

    if min(h, w) < 256:
        reasons.append("LOW_RESOLUTION")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_val = _laplacian_var(gray)
    debug["blur_lap_var"] = blur_val
    if blur_val < float(th["blur_min"]):
        reasons.append("BLUR")

    over_r, under_r = _exposure_ratios(gray)
    debug["overexpose_ratio"] = over_r
    debug["underexpose_ratio"] = under_r
    if over_r > float(th["overexpose_ratio_max"]):
        reasons.append("OVEREXPOSED")
    if under_r > float(th["underexpose_ratio_max"]):
        reasons.append("UNDEREXPOSED")

    cast = _color_cast_score(image_bgr)
    debug["color_cast_score"] = cast
    if cast > float(th["color_cast_max"]):
        reasons.append("COLOR_CAST")

    # ---- v2: rough tongue presence / size (heuristic) ----
    do_rough = cast <= float(th["color_cast_max"]) * 1.25
    debug["rough_enabled"] = do_rough

    if do_rough:
        mask = _rough_tongue_mask(image_bgr)
        tongue_ratio = float((mask > 0).mean())
        debug["rough_tongue_area_ratio"] = tongue_ratio

        suspect_th = float(th.get("tongue_area_ratio_suspect", 0.06))
        min_th = float(th.get("tongue_area_ratio_min", 0.15))

        if tongue_ratio < suspect_th:
            reasons.append("NO_TONGUE_SUSPECTED")
        elif tongue_ratio < min_th:
            reasons.append("TONGUE_TOO_SMALL")

        bin_mask = ((mask > 0).astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        comp_count = max(0, num_labels - 1)
        debug["rough_component_count"] = comp_count
        if comp_count >= 8 and tongue_ratio > 0.02:
            reasons.append("OCCLUSION_OR_BACKGROUND")

    reasons = list(dict.fromkeys(reasons))
    passed = len(reasons) == 0
    if passed:
        tip = "图片质量合格。"
    else:
        tips = [DEFAULT_REASON_TIPS.get(r, "请重新拍摄。") for r in reasons[:2]]
        tip = "；".join(tips)

    return QualityResult(passed=passed, reason_codes=reasons, user_tip=tip, debug=debug)

if __name__ == "__main__":
    import sys
    cfg = load_config()
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    out = quality_gate(img, cfg)
    print(out)
