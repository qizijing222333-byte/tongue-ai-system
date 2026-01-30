import cv2
from inference.quality_gate import load_config, quality_gate

cfg = load_config()

img = cv2.imread("sample.jpg")
if img is None:
    raise FileNotFoundError("找不到 sample.jpg：请把一张测试图片放在项目根目录并命名为 sample.jpg")

res = quality_gate(img, cfg)

print("pass:", res.passed)
print("reasons:", res.reason_codes)
print("tip:", res.user_tip)
print("debug:", res.debug)
