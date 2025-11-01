# Head Pose Estimation using Machine Learning & Mediapipe

## 📌 Project Overview
This project implements a complete Head Pose Estimation pipeline that predicts pitch, yaw, and roll angles from image or video input.
It combines **Mediapipe FaceMesh** for facial landmark extraction with Machine Learning **Regression Models** (SVR & Random Forest) for accurate head pose prediction.
The system is fully production-ready with **FastAPI** deployment and **Docker** support.
## 🚀 Features
- Extracts facial landmarks using **Mediapipe**  
- Preprocesses data for ML models  
- Trains and evaluates regression models (SVR & RF)  
- Achieved strong performance:  
  - Pitch → **MSE: 62.36, R²: 0.726**  
  - Yaw → **MSE: 116.87, R²: 0.866**  
  - Roll → **MSE: 36.65, R²: 0.884**  
- Saves models with **joblib** for deployment  
- Head pose visualization with 3D axes drawn on the face
- Supports images, videos, and API-based inference
- Deployed as a REST API with FastAPI
- Docker image available for easy deployment 

## 🛠️ Tech Stack
- **Python**  
- **Mediapipe** (FaceMesh Landmark Extraction)  
- **Scikit-learn** (SVR, Random Forest)  
- **OpenCV** (Image/video processing & visualization)  
- **Joblib** (Model Persistence)
- **Fast API** (Model Serving)
- **Docker** (Model Containerization & Deployment)

## 📁 Project Structure
```bash
Head-Pose-Estimation/
│
├── main.py                         # FastAPI entry point (API server)
├── head_pose_estimation.py         # Core processing: face detection + pose estimation
├── requirements.txt
├── dockerfile
├── README.md
│
├── models/                         # ML models (SVR/RF for yaw/pitch/roll)
│   ├── svr_pitch.joblib
│   ├── rf_yaw.joblib
│   └── svr_roll.joblib
│
├── static/                         # Frontend
│   └── index.html
│
├── samples/                        # Sample images & videos
│   ├── test_image.png
│   └── test_video.mp4

```
## ⚡ How to Run
1. **Clone the repository**  
   ```bash
   git clone https://github.com/HabibaMAtiia/head-pose-estimation.git
   cd head-pose-estimation
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run inference on an image or video**
   ```bash
   python main.py

## 🔮 Future Improvements

- Add real-time head pose estimation using webcam + FastAPI WebSockets

- Deploy Docker image to cloud platforms (Render, Railway, AWS)

## 👩‍💻 Author
**Habiba M. Attia**
