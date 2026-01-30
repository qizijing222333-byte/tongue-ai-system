from __future__ import annotations
from typing import Any, Dict, Optional
import os
import cv2
import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None


def load_model_cfg(path: str = "configs/model.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fake_predictor() -> Dict[str, Any]:
    # 没有权重时，先保证系统可用（你现在的规则/网页不崩）
    return {
        "body_color": {"label": "pale", "prob": 0.78, "pale": 0.78, "red": 0.15, "dark": 0.07},
        "coat_color": {"label": "white", "prob": 0.80, "white": 0.80, "yellow": 0.12, "no_coat": 0.08},
        "thickness": {"label": "thick", "prob": 0.70, "thin": 0.30, "thick": 0.70},
        "greasy": {"label": "greasy", "prob": 0.66, "non_greasy": 0.34, "greasy": 0.66},
        "features": {"tooth_mark": {"prob": 0.0}, "fissure": {"prob": 0.0}},
    }


class TonguePredictor:
    """
    推理接口壳：后面不管你换成 Torch/ONNX/TFLite，都保持 predict(image_bgr)->dict 输出一致。
    """
    def __init__(self, model_cfg: Dict[str, Any]):
        self.cfg = model_cfg
        self.backend = model_cfg.get("backend", "torch")
        self.device = model_cfg.get("device", "cpu")
        self.input_size = int(model_cfg.get("input_size", 224))
        self.weights_path = model_cfg.get("weights_path", "")

        self.labels = model_cfg["labels"]
        self.feature_cfg = model_cfg.get("features", {})

        self._model = None
        self._ready = False
        self.weights_path = ""
        self.model_version = "FAKE-0.0"
        # 目前先只搭 torch 壳
        if self.backend == "torch" and torch is not None and self.weights_path and os.path.exists(self.weights_path):
            self._load_torch()
        else:
            # 没权重 / 没torch：fallback
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def _load_torch(self) -> None:
        # 这里先假设你训练后保存的是 torchscript 或 state_dict 都可
        # 最省事：训练脚本导出 torchscript: torch.jit.script(model).save(...)
        self._model = torch.jit.load(self.weights_path, map_location=self.device)
        self._model.eval()
        self._ready = True

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        # v0：先直接 resize；后面再接“检测/分割 ROI crop + 颜色校正”
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        mean = np.array(self.cfg["normalize"]["mean"], dtype=np.float32)
        std = np.array(self.cfg["normalize"]["std"], dtype=np.float32)
        img = (img - mean) / std

        # CHW
        img = np.transpose(img, (2, 0, 1))
        return img

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if not self._ready:
            return _fake_predictor()

        x = self._preprocess(image_bgr)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 约定：模型 forward 输出 dict（训练时我们也按这个约定写）
            out = self._model(x)

        # 将模型输出整理成你规则引擎需要的结构
        return self._postprocess(out)

    def _postprocess(self, out: Dict[str, Any]) -> Dict[str, Any]:
        # out 约定:
        # out["body_color_logits"] shape [1,3]
        # out["coat_color_logits"] shape [1,3]
        # out["thickness_logits"] shape [1,2]
        # out["greasy_logits"] shape [1,2]
        # 可选: out["tooth_mark_logit"], out["fissure_logit"] shape [1,1]
        def softmax_dict(logits, names):
            p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
            probs = {names[i]: float(p[i]) for i in range(len(names))}
            label = max(probs, key=probs.get)
            return {"label": label, "prob": float(probs[label]), **probs}

        pred = {
            "body_color": softmax_dict(out["body_color_logits"], self.labels["body_color"]),
            "coat_color": softmax_dict(out["coat_color_logits"], self.labels["coat_color"]),
            "thickness": softmax_dict(out["thickness_logits"], self.labels["thickness"]),
            "greasy": softmax_dict(out["greasy_logits"], self.labels["greasy"]),
            "features": {
                "tooth_mark": {"prob": 0.0},
                "fissure": {"prob": 0.0},
            }
        }

        # 可选特征（二分类 sigmoid）
        if "tooth_mark_logit" in out:
            pred["features"]["tooth_mark"]["prob"] = float(torch.sigmoid(out["tooth_mark_logit"]).item())
        if "fissure_logit" in out:
            pred["features"]["fissure"]["prob"] = float(torch.sigmoid(out["fissure_logit"]).item())

        return pred
