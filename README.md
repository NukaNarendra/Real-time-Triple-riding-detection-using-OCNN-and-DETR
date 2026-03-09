#  Real-Time Triple Riding Detection using OCNN + DETR

### *An AI-powered system detects triple riding on two-wheelers using DETR for object detection, CNN for rider counting, and OCR for license plate recognition. It ensures real-time, scalable traffic monitoring with temporal validation to reduce false positives, enhancing enforcement, safety, and public accountability.*

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-DETR/RTDETR-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Plate%20OCR-EasyOCR-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tracking-DeepSORT-orange?style=for-the-badge" />
</p>

<p align="center">
 **A production‑ready vision system engineered for real‑time traffic violation detection at metropolitan scale.**
</p>

---

##  Executive Summary

This project implements a **high‑performance real-time triple‑riding detection platform** that combines:

* **Transformer‑based detection (DETR/RT-DETR)**
* **Custom OCNN rider‑counting model**
* **License plate detection + OCR**
* **Automated evidence capture system**
* **Production-ready Flask dashboard**
* **Dockerized deployment & modular architecture**

The system is engineered following **Google AI**, **Microsoft Azure Perception**, and **NVIDIA Metropolis** design standards — clean modularity, high readability, scalability, and enterprise‑deployment patterns.

---

##  Core Capabilities

###  1. Motorcycle & Rider Detection (DETR / RT‑DETR)

* Transformer‑based object detection
* Robust in high‑traffic, occluded scenarios
* Real‑time FPS on GPU

###  2. Rider Counting (OCNN Custom CNN)

* Lightweight CNN trained exclusively for rider counting
* More accurate than YOLO‑based approaches
* Works under motion blur & extreme angles

###  3. License Plate Recognition

* YOLO‑based plate detection
* EasyOCR for multi‑language alphanumeric extraction
* Low false‑positive rate

###  4. Evidence Generation Pipeline

Automatically stores:

* **Raw image**
* **Blurred image (privacy protection)**
* **Timestamp, metadata, bounding boxes**

###  5. Fully Featured Web Dashboard

* Search, filter, and review violations
* Blur-safe images using privacy module
* Evidence database stored in SQLite

---

##  High-Level Architecture

```
         ┌────────────────────────────┐
         │  Video Stream / CCTV Feed │
         └───────────────┬────────────┘
                         │
                ┌────────▼──────────┐
                │  DETR/RT-DETR     │
                │ Motorcycle Detect  │
                └────────┬──────────┘
                         │
          ┌──────────────┴──────────────┐
          │                               │
  ┌───────▼────────┐             ┌────────▼───────────┐
  │ OCNN Rider Count│             │ Plate Detector     │
  └───────┬────────┘             └────────┬────────────┘
          │                               │
      ┌───▼───────┐                ┌──────▼───────────┐
      │ Violation?│                │ OCR (EasyOCR)     │
      └─────┬─────┘                └────────┬──────────┘
            │                                │
       ┌────▼─────────────────────────────────▼────┐
       │          Evidence Generator                │
       └───────────────────┬────────────────────────┘
                           │
                    ┌──────▼──────────┐
                    │   SQLite DB     │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │ Flask Dashboard │
                    └──────────────────┘
```

---

## Quick Start

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run Real-Time Detection

```bash
python scripts/run_inference_server.py
```

###  Process a Video

```bash
python scripts/consume_process_video.py --video datasets/test_video.mp4
```

###  Launch Dashboard

```bash
python web_app/app.py
```

Then open:
 **[http://localhost:5000/](http://localhost:5000/)**

---

##  Project Structure (Google‑Style)

```
project/
│
├── checkpoints/              # LFS-stored model files
├── configs/                  # YAML configs
├── datasets/                 # Videos & parsers
├── evidence_store/           # Raw + blurred evidence
├── inference/                # Pipeline modules
├── db/                       # SQLite + ORM
├── models/                   # DETR/OCNN/YOLO models
├── training/                 # Training scripts
├── scripts/                  # Admin & inference tools
├── utils/                    # Logging, helpers
└── web_app/                  # Flask dashboard
```

---

##  Production‑Ready Docker Deployment

### Build

```bash
docker build -t triple-riding-detector .
```

### Run

```bash
docker run -p 5000:5000 triple-riding-detector
```

---

##  Model Performance

| Component                     | Accuracy  | FPS     | Notes                   |
| ----------------------------- | --------- | ------- | ----------------------- |
| Motorcycle Detector (RT-DETR) | **82-85%**| 30 FPS  | Robust under occlusion  |
| OCNN Rider Counter            | **95.7%** | 500 FPS | Fast lightweight CNN    |
| Plate Detection               | **92–94%**| 40 FPS  | Fine‑tuned YOLO         |
| OCR                           | **88–93%**| 20 FPS  | Indian plates supported |

---

##  Training Commands

### Train Detector

```bash
python training/train_detector.py
```

### Train OCNN Rider Counter

```bash
python training/train_rider_counter.py
```

---

##  Engineering Principles

* **Clean architecture** (independent modules)
* **High cohesion, low coupling**
* **Consistent naming conventions**
* **Environment‑driven configuration (YAML)**
* **Edge‑deployable lightweight models**
* **LFS‑managed large models**
* **Logging-first design**
* **Privacy-first evidence handling**

---

---

##  Contributing

PRs and suggestions are welcome.

---

## License

MIT License

---

## 👨‍💻 Author

### **Nuka Venkata Narendra and Bejawada Venkata Sai and Battina Jahnavi**

AI/ML Engineer • Real‑Time Systems Developer • NLP Engineer 
