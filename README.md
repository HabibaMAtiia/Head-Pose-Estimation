# Head Pose Estimation using Machine Learning & Mediapipe

## 📌 Project Overview
This project implements a **head pose estimation system** that predicts the **pitch, yaw, and roll angles** of a face from video input.  
It leverages **Mediapipe FaceMesh** for landmark extraction and machine learning models (**SVR & Random Forest**) for regression.  
Models are trained, evaluated, and saved using **joblib** for production-ready deployment.

## 🚀 Features
- Extracts facial landmarks using **Mediapipe**  
- Preprocesses data for ML models  
- Trains and evaluates regression models (SVR & RF)  
- Achieved strong performance:  
  - Pitch → **MSE: 62.36, R²: 0.726**  
  - Yaw → **MSE: 116.87, R²: 0.866**  
  - Roll → **MSE: 36.65, R²: 0.884**  
- Saves models with **joblib** for deployment  
- Annotates head pose axes on video and saves the output  

## 🛠️ Tech Stack
- **Python**  
- **Mediapipe** (FaceMesh for landmarks)  
- **Scikit-learn** (SVR, Random Forest)  
- **OpenCV** (video processing & visualization)  
- **Joblib** (model persistence)
- **Fast API** (model deployment)


## ⚡ How to Run
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/head-pose-estimation.git
   cd head-pose-estimation
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run inference on an image or video**
   ```bash
   python main.py
   
