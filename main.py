from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
from pathlib import Path
import mimetypes
from head_pose_estimation import process_image, process_video

app = FastAPI(title="Head Pose Estimation API", version="1.0.0")

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

def get_file_type(filename: str) -> str:
    """
    Determine if file is image or video based on extension
    
    Args:
        filename (str): Name of the file
    
    Returns:
        str: 'image', 'video', or 'unknown'
    """
    extension = Path(filename).suffix.lower()
    
    if extension in IMAGE_EXTENSIONS:
        return 'image'
    elif extension in VIDEO_EXTENSIONS:
        return 'video'
    else:
        return 'unknown'

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Head Pose Estimation API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "Upload image or video for head pose estimation",
            "/health": "Health check endpoint"
        },
        "supported_formats": {
            "images": list(IMAGE_EXTENSIONS),
            "videos": list(VIDEO_EXTENSIONS)
        }
    }


@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """
    Process uploaded image or video for head pose estimation
    
    Args:
        file: Uploaded image or video file
    
    Returns:
        FileResponse: Processed file with head pose visualization
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Determine file type
    file_type = get_file_type(file.filename)
    
    if file_type == 'unknown':
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}"
        )
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate output filename
        file_stem = Path(file.filename).stem
        file_ext = Path(file.filename).suffix
        
        if file_type == 'image':
            output_filename = f"{file_stem}_pose{file_ext}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Process image
            result = process_image(input_path, display=False, save_path=output_path)
            
            if result is None:
                raise HTTPException(status_code=422, detail="Could not process image. Please ensure it's a valid image file.")
            
            img, pitch, yaw, roll = result
            
            # Set response headers with pose information
            headers = {
                "X-Pitch": str(round(pitch, 2)) if pitch is not None else "0.0",
                "X-Yaw": str(round(yaw, 2)) if yaw is not None else "0.0", 
                "X-Roll": str(round(roll, 2)) if roll is not None else "0.0",
                "X-File-Type": "image"
            }
            
        else:  # video
            output_filename = f"{file_stem}_pose{file_ext}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Process video
            success = process_video(input_path, output_path)
            
            if not success:
                raise HTTPException(status_code=422, detail="Could not process video. Please ensure it's a valid video file.")
            
            headers = {
                "X-File-Type": "video",
                "X-Processing": "completed"
            }
        
        # Check if output file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to generate processed file")
        
        # Return processed file
        return FileResponse(
            path=output_path,
            filename=output_filename,
            headers=headers,
            media_type=mimetypes.guess_type(output_path)[0]
        )
    
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting Head Pose Estimation API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)