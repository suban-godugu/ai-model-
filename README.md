# WaferVision-AI

AI-powered semiconductor wafer defect analysis system using Deep Learning, Computer Vision, FastAPI, and Streamlit for intelligent wafer inspection and spatial defect analytics.

---

# Overview

WaferVision-AI is an industry-oriented semiconductor wafer inspection system designed to automate defect detection and wafer condition analysis using Artificial Intelligence.

The project combines:

* CNN-based wafer defect classification
* Grad-CAM visualization
* FastAPI backend deployment
* Streamlit interactive dashboard
* Spatial defect analytics

This system helps improve semiconductor manufacturing quality by reducing manual inspection effort and enabling intelligent defect analysis.

---

# Features

## Wafer Defect Classification

Classifies wafer defects into multiple categories:

* Center
* Donut
* Edge-Loc
* Edge-Ring
* Local
* Near-Full
* Normal
* Random
* Scratch

---

## Grad-CAM Visualization

Provides explainable AI visualization to highlight important defect regions influencing model predictions.

---

## FastAPI Backend

REST API integration for real-time wafer prediction and deployment.

---

## Streamlit Dashboard

Interactive dashboard for:

* Image upload
* Prediction visualization
* Heatmap display
* Defect analysis
* AI prediction monitoring

---

## U-Net Segmentation (In Progress)

Proposed U-Net based segmentation pipeline for future pixel-level defect localization and spatial defect analysis.

---

# Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| PyTorch      | Deep Learning framework   |
| FastAPI      | Backend API development   |
| Streamlit    | Interactive dashboard     |
| OpenCV       | Image processing          |
| NumPy        | Numerical computations    |
| Matplotlib   | Visualization             |
| Scikit-learn | Model evaluation          |

---

# Project Architecture

Wafer Image
↓
Preprocessing
↓
CNN Classification
↓
Grad-CAM Visualization
↓
Dashboard/API Output

---

# Model Capabilities

## CNN Classification Model

* Detects wafer defect categories
* Performs automated wafer condition analysis
* Generates intelligent defect predictions

---

## Spatial Defect Analysis

* Visualizes defect-prone regions
* Generates heatmap-based analysis
* Supports future segmentation integration

---

# Folder Structure

WaferVision-AI/

├── src/
├── models/
├── temp/
├── LICENSE
├── requirements.txt
├── README.md

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
```

## Create Virtual Environment

```bash
py -3.11 -m venv venv
```

## Activate Virtual Environment

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run FastAPI Backend

```bash
python -m uvicorn src.api:app --reload
```

Backend URL:

```bash
http://127.0.0.1:8000
```

---

# Run Streamlit Dashboard

```bash
streamlit run src/dashboard.py
```

---

# Future Enhancements

* Defect Severity Scoring
* Lot-Level Wafer Analytics
* Root Cause Prediction
* U-Net Defect Segmentation
* Hybrid Parametric + Vision AI
* Real-Time Factory Monitoring
* Yield Prediction System

---

# Applications

* Semiconductor Manufacturing
* Automated Optical Inspection
* Smart Manufacturing
* Industrial Quality Control
* AI-Based Defect Detection

---

# License

MIT License

Copyright (c) 2026 Keerthan G V, Verilumen Labs

---

# Developed By

Keerthan G V
Verilumen Labs

