# Tongue AI API Spec (v0.1)

## Request
POST /api/predict
- Content-Type: multipart/form-data
- form-data:
  - image: file (jpg/png) **required**
  - patient_id: string (optional)
  - return_debug: bool (optional, default=false)

## Response (JSON)

### 0) Meta
- request_id: string
- timestamp: string (ISO8601)
- model_version: string
- rule_version: string

### 1) Quality Gate
- quality:
  - pass: bool
  - reason_codes: string[]   # e.g. ["BLUR","OVEREXPOSED","NO_TONGUE_DETECTED"]
  - user_tip: string         # user-friendly guidance

If quality.pass == false:
- detect: optional
- segment: optional
- tongue: null
- constitution: null
- artifacts: optional

### 2) Detection (Tongue Exists + Location)
- detect:
  - found: bool
  - conf: float (0~1)
  - bbox: [x1,y1,x2,y2]  # normalized 0~1, relative to original image

### 3) Segmentation (Tongue Mask Quality)
- segment:
  - conf: float (0~1)
  - tongue_area_ratio: float (0~1)  # mask area / roi area

### 4) Tongue Labels (Model Outputs)
- tongue:
  - body_color: {label: "pale"|"red"|"dark", prob: float}
  - coat_color: {label: "white"|"yellow"|"no_coat", prob: float}
  - thickness:  {label: "thin"|"thick", prob: float}
  - greasy:     {label: "non_greasy"|"greasy", prob: float}
  - features:
      tooth_mark: {prob: float}
      fissure: {prob: float}

### 5) Constitution (Rule Engine Outputs)
- constitution:
  - decision: "CLEAR" | "MIXED" | "LOW_CONF"
  - topk: [
      {
        name: "phlegm_damp" | "damp_heat" | "yin_def",
        score: float,       # 0~100
        confidence: float,  # 0~1
        evidence: string,
        advice: {
          diet: string,
          lifestyle: string,
          exercise: string,
          medical: string
        }
      }
    ]

### 6) Artifacts (Optional)
- artifacts:
  - roi_image_url: string | null
  - mask_image_url: string | null
  - debug_images_url: string[] | null

## Notes
- This system provides decision support and is not a medical diagnosis.
- If quality is LOW or evidence is insufficient, return constitution.decision="LOW_CONF".
