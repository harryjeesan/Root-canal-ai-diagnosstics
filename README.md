<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Root Canal AI Diagnostics

An AI-powered system for the automated detection of endodontic failures from dental radiographs. This project implements advanced image enhancement via Contourlet Transform filtering and a hierarchical Knowledge Distillation pipeline to achieve high-precision results on lightweight models.

View the app in AI Studio: https://ai.studio/apps/drive/1n8QXhbFSpQTjHTEeTUvkUO1dAyRZwbcD

## 🚀 Features

- **Image Enhancement:** Custom Contourlet Transform and Filter Fusion (NLM + Bayesian) to improve edge visibility in dental X-rays.
- **Knowledge Distillation:** A "Model Ladder" architecture that transfers "dark knowledge" from a large Teacher (YOLOv8m) to an ultra-efficient Student (YOLOv5n).
- **High-Fidelity Labeling:** Integration with Grounded SAM (GSAM) for automated, high-precision bounding box generation.
- **Explainable AI:** Integrated Grad-CAM heatmaps to visualize model attention areas for clinical trustworthiness.
- **Real-time Web Interface:** React-based frontend with a Flask backend for instant diagnostic feedback.

## 🛠️ Tech Stack

- **Machine Learning:** YOLOv8 (Ultralytics), YOLOv5, PyTorch, OpenAI CLIP (Auto-labeling)
- **Image Processing:** OpenCV, SciPy, Scikit-image (Custom Contourlet implementation)
- **Frontend:** React 19, Vite, TypeScript, TailwindCSS
- **Backend:** Python 3.12, Flask API

## 📦 Getting Started

### Prerequisites

- Node.js (for frontend)
- Python 3.9+
- NVIDIA GPU (Recommended for training)

```bash
pip install -r requirements.txt
npm install
```

### Run Locally

1. **Configure Environment:** Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key.

2. **Preprocess Dataset (Contourlet Filter):** Apply the directional filter bank to enhance curvilinear structures.

   ```bash
   python preprocess_dataset.py --backup
   ```

3. **Run Training Pipeline:** Orchestrate the full Teacher-Student distillation process.

   ```bash
   python run_distillation_pipeline.py --stage all
   ```

4. **Launch the Application:** Start both servers to begin diagnostics.

   ```bash
   # Terminal 1: Backend Inference Server
   python inference_server.py

   # Terminal 2: Frontend React App
   npm run dev
   ```

## 📊 Model Architecture & Results

| Model | Parameters | mAP50 (Filtered) | Precision |
|---|---|---|---|
| Teacher (YOLOv8m) | 25.9M | ~0.87 | 0.88 |
| Student Distilled (YOLOv5n) | 2.5M | 0.99 (Goal) | 99.16% |
| Student Baseline (YOLOv8n) | 6.9M | 0.00 (Untrained) | 0.00 |

*Data based on findings from the research paper: "Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis".*

## 🧪 Filtering Methods Implemented

- `apply_mean_filter`: Uses `cv2.blur`
- `apply_median_filter`: Uses `cv2.medianBlur`
- `apply_gaussian_filter`: Uses `cv2.GaussianBlur`
- `apply_non_local_means`: Uses `cv2.fastNlMeansDenoising`
- `apply_bayesian_wavelet`: Uses `skimage.restoration.denoise_wavelet` (BayesShrink)
- `apply_contourlet_proxy`: Uses `cv2.getGaborKernel` and convolution (Gabor Filter)

## 📁 Project Structure

- `knowledge_distillation.py` — Custom loss and trainer for knowledge transfer
- `hierarchical_distillation.py` — Implementation of the 3-stage model ladder (v8m → v8n → v5n)
- `contourlet_filter.py` — Directional image decomposition algorithms
- `filter_fusion.py` — Strategy combining NLM, Bayesian, and Contourlet filters
- `inference_server.py` — Flask API serving multi-model ensembles with real-time filtering
