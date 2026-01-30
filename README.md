# tongue-ai-system
# Tongue AI System (Deep Learning + Rule Engine)

A tongue image analysis system for:
- Tongue body / coating classification (multi-task)
- Tongue feature recognition (tooth mark, fissure)
- Rule-engine constitution inference (Top-2): damp-heat / phlegm-damp / yin-def
- Patient report generation
- Web UI + Raspberry Pi offline deployment

## Pipeline
Quality Gate -> Tongue Detection -> ROI Crop -> Tongue Segmentation -> Color-preserving ROI ->
Multi-task Classification -> Rule Engine (Top-2 constitution) -> Report

## Repository Structure
- configs/: global configs (labels, thresholds, paths, rules)
- docs/: API spec, design docs
- data_pipeline/: schema, preprocessing, augmentation, dataset split
- models/: detector, segmenter, multitask classifier
- train/: training scripts
- eval/: metrics, confusion matrix, error analysis
- export/: model export + quantization
- inference/: end-to-end inference pipeline
- rule_engine/: rules_v2.json + scoring + advice templates
- web_app/: backend API + UI
- edge_rpi/: Raspberry Pi camera + offline inference
- utils/: shared utilities

## Data & Weights
Dataset, annotations, logs, and model weights are NOT committed to GitHub.
See `.gitignore` and keep them under `data/`.

## Quick Start (placeholder)
- Create venv / conda env
- Install dependencies (to be added)
- Run web API (to be added)
- Run Raspberry Pi client (to be added)

## Disclaimer
This tool provides decision support and is not a medical diagnosis.
