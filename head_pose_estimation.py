import cv2
import numpy as np
import mediapipe as mp
import joblib
import matplotlib.pyplot as plt

# Load trained models
svr_pitch = joblib.load("svr_pitch.joblib")
rf_yaw = joblib.load("rf_yaw.joblib")
svr_roll = joblib.load("svr_roll.joblib")

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def preprocess(face, width, height):
    """
    Preprocess face landmarks for model prediction
    
    Args:
        face: MediaPipe face landmarks
        width: Image width
        height: Image height
    
    Returns:
        tuple: (x_coordinates, y_coordinates) normalized
    """
    x_val = [lm.x * width for lm in face.landmark]
    y_val = [lm.y * height for lm in face.landmark]

    x_val = np.array(x_val) - np.mean(x_val[1])
    y_val = np.array(y_val) - np.mean(y_val[1])

    max_val = max(abs(x_val).max(), abs(y_val).max())
    x_val = x_val / max_val if max_val != 0 else x_val
    y_val = y_val / max_val if max_val != 0 else y_val
    
    return x_val, y_val

def draw_axes(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    """
    Draw 3D pose axes on image
    
    Args:
        img: Input image
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        tdx: X coordinate of center point
        tdy: Y coordinate of center point
        size: Length of axes lines
    
    Returns:
        Image with drawn axes
    """
    # Convert to radians
    pitch = np.radians(pitch)
    yaw = np.radians(-yaw)
    roll = np.radians(roll)
    
    h, w = img.shape[:2]
    if tdx is None: 
        tdx = w // 2
    if tdy is None: 
        tdy = h // 2

    # Calculate axes endpoints
    # X axis (red) – yaw
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y axis (green) – pitch
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z axis (blue) – roll
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # Draw axes lines
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)  # X - red
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)  # Y - green
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z - blue
    
    return img

def process_image(image_path, display=True, save_path=None):
    """
    Process a single image for head pose estimation
    """
    # Load image in BGR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb.copy())
    
    pitch_pred, yaw_pred, roll_pred = None, None, None
    img_with_axes = img_bgr.copy()   # keep this in BGR for OpenCV drawing
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Preprocess landmarks
            pre_x, pre_y = preprocess(face_landmarks, width=img_bgr.shape[1], height=img_bgr.shape[0])
            
            # Prepare features
            features = np.array([[val for xy in zip(pre_x, pre_y) for val in xy]])
            
            # Predict head pose
            pitch_pred = svr_pitch.predict(features)[0]
            yaw_pred = rf_yaw.predict(features)[0]
            roll_pred = svr_roll.predict(features)[0]
            
            # Nose landmark for axes origin
            h, w, _ = img_bgr.shape
            nose = face_landmarks.landmark[4]
            nose_tdx, nose_tdy = int(nose.x * w), int(nose.y * h)
            
            # Draw axes on BGR image
            img_with_axes = draw_axes(img_with_axes, yaw_pred, pitch_pred, roll_pred,
                                      nose_tdx, nose_tdy, size=80)
            
            title_text = f"Predicted Pose: Pitch={pitch_pred:.2f}°, Yaw={yaw_pred:.2f}°, Roll={roll_pred:.2f}°"
            break
    else:
        # No face detected → fallback
        h, w = img_bgr.shape[:2]
        img_with_axes = draw_axes(img_with_axes, 0, 0, 0, w//2, h//2, size=80)
        title_text = "No face detected. Using default pose."
    
    # ✅ Display result (convert to RGB for matplotlib)
    if display:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_with_axes, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title_text)
        plt.show()
    
    # ✅ Save result (BGR is correct for OpenCV write)
    if save_path:
        cv2.imwrite(save_path, img_with_axes)
        print(f"Image saved to: {save_path}")
    
    return img_with_axes, pitch_pred, yaw_pred, roll_pred


def process_video(video_path, output_path):
    """
    Process a video for head pose estimation and save the result
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path where the processed video will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return False
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Preprocess landmarks
            pre_x, pre_y = preprocess(face_landmarks, frame.shape[1], frame.shape[0])
            
            # Combine features
            features = []
            for i in range(len(pre_x)):
                features.extend([pre_x[i], pre_y[i]])
            features = np.array(features).reshape(1, -1)
            
            # Predict pose angles
            pitch_pred = svr_pitch.predict(features)[0]
            yaw_pred = rf_yaw.predict(features)[0]
            roll_pred = svr_roll.predict(features)[0]
            
            # Get nose landmark position
            nose = face_landmarks.landmark[4]
            nose_tdx = int(nose.x * frame.shape[1])
            nose_tdy = int(nose.y * frame.shape[0])
            
            # Draw axes on frame
            frame = draw_axes(frame, yaw_pred, pitch_pred, roll_pred, 
                            nose_tdx, nose_tdy, size=80)
            
            # Add text overlay with angles
            cv2.putText(frame, f"P: {pitch_pred:.1f}, Y: {yaw_pred:.1f}, R: {roll_pred:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Progress update every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Video processing complete! Output saved to: {output_path}")
    return True