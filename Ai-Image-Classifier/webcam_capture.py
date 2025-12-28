"""
Webcam Image Capture Tool for Dataset Collection
=================================================
This script helps you collect your own facial expression dataset using your webcam.

Usage:
    python webcam_capture.py

Controls:
    - Press number keys (1-5) to capture image for that emotion
    - Press 'q' to quit
    - Press 's' to switch camera (if you have multiple cameras)
"""

import cv2
import os
from datetime import datetime
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define your emotion categories
# You can customize these to whatever expressions you want to capture
EMOTIONS = {
    '1': 'happy',
    '2': 'sad',
    '3': 'neutral',
    '4': 'angry',
    '5': 'surprised'
}

# Where to save captured images
SAVE_DIR = 'data/train'

# Image settings
IMG_WIDTH = 224   # Width of saved images
IMG_HEIGHT = 224  # Height of saved images

# Capture settings
CAPTURE_DELAY = 0.5  # Seconds to wait between captures (prevents accidental duplicates)

# Display settings
WINDOW_NAME = 'Facial Expression Capture - Press 1-5 to capture, Q to quit'
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


# ==============================================================================
# SETUP FUNCTIONS
# ==============================================================================

def create_emotion_folders():
    """
    Create folders for each emotion category.
    
    Creates:
    --------
    data/train/happy/
    data/train/sad/
    data/train/neutral/
    etc.
    """
    
    print("\n" + "="*70)
    print("SETTING UP EMOTION FOLDERS")
    print("="*70)
    
    for emotion in EMOTIONS.values():
        emotion_path = os.path.join(SAVE_DIR, emotion)
        os.makedirs(emotion_path, exist_ok=True)
        print(f"✓ Created folder: {emotion_path}")
    
    print("="*70 + "\n")


def count_existing_images():
    """
    Count how many images already exist in each emotion folder.
    
    Returns:
        dict: Emotion name -> count
    """
    
    counts = {}
    
    for emotion in EMOTIONS.values():
        emotion_path = os.path.join(SAVE_DIR, emotion)
        if os.path.exists(emotion_path):
            # Count image files
            images = [f for f in os.listdir(emotion_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[emotion] = len(images)
        else:
            counts[emotion] = 0
    
    return counts


# ==============================================================================
# WEBCAM FUNCTIONS
# ==============================================================================

def initialize_camera(camera_id=0):
    """
    Initialize the webcam.
    
    Args:
        camera_id (int): Camera device ID (0 for default camera)
        
    Returns:
        cv2.VideoCapture: Camera object
    """
    
    print("Initializing camera...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Make sure your webcam is connected.")
    
    # Set camera resolution (optional, adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Camera initialized successfully")
    
    return cap


def preprocess_face(frame):
    """
    Preprocess the captured frame for better face detection.
    
    Optional preprocessing:
    - Convert to grayscale for face detection
    - Equalize histogram for better contrast
    
    Args:
        frame: Original BGR frame from webcam
        
    Returns:
        frame: Preprocessed frame
    """
    
    # You can add preprocessing here if needed
    # For now, we'll just return the original frame
    return frame


def detect_face(frame, face_cascade):
    """
    Detect faces in the frame using Haar Cascade.
    
    Args:
        frame: Input image frame
        face_cascade: OpenCV face detector
        
    Returns:
        tuple: (face_found, face_region, face_coordinates)
    """
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)  # Minimum face size
    )
    
    if len(faces) > 0:
        # Get the largest face (in case multiple detected)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        return True, face_region, (x, y, w, h)
    
    return False, None, None


def draw_ui(frame, counts, last_emotion=None, countdown=None):
    """
    Draw user interface on the frame.
    
    Shows:
    - Instructions
    - Image counts per emotion
    - Countdown timer (if capturing)
    - Face detection rectangle
    
    Args:
        frame: Frame to draw on
        counts: Dictionary of image counts per emotion
        last_emotion: Last captured emotion (for feedback)
        countdown: Countdown value (if capturing)
        
    Returns:
        frame: Frame with UI drawn
    """
    
    # Create a copy to draw on
    display_frame = frame.copy()
    
    # Get frame dimensions
    height, width = display_frame.shape[:2]
    
    # Background for text (semi-transparent)
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    # Draw instructions
    y_offset = 30
    cv2.putText(display_frame, "FACIAL EXPRESSION CAPTURE TOOL", 
                (10, y_offset), FONT, 0.8, (255, 255, 255), FONT_THICKNESS)
    
    y_offset += 40
    cv2.putText(display_frame, "Press number to capture:", 
                (10, y_offset), FONT, FONT_SCALE, (200, 200, 200), 1)
    
    # Draw emotion keys
    y_offset += 30
    for key, emotion in EMOTIONS.items():
        count = counts.get(emotion, 0)
        color = (0, 255, 0) if count >= 50 else (100, 100, 255)
        text = f"[{key}] {emotion.capitalize()}: {count} images"
        cv2.putText(display_frame, text, (10, y_offset), 
                   FONT, 0.6, color, 1)
        y_offset += 25
    
    # Draw additional instructions
    y_offset += 10
    cv2.putText(display_frame, "[Q] Quit  [S] Switch Camera", 
                (10, y_offset), FONT, 0.5, (150, 150, 150), 1)
    
    # Show last captured emotion
    if last_emotion:
        cv2.putText(display_frame, f"Captured: {last_emotion}!", 
                   (width - 250, height - 20), FONT, 0.7, (0, 255, 0), 2)
    
    # Show countdown if capturing
    if countdown is not None:
        text = f"Capturing in {countdown}..."
        text_size = cv2.getTextSize(text, FONT, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(display_frame, text, (text_x, text_y), 
                   FONT, 2, (0, 255, 255), 3)
    
    return display_frame


def save_image(face_region, emotion):
    """
    Save the captured face image to the appropriate emotion folder.
    
    Args:
        face_region: Cropped face image
        emotion: Emotion category name
        
    Returns:
        str: Path to saved image
    """
    
    # Resize face to standard size
    face_resized = cv2.resize(face_region, (IMG_WIDTH, IMG_HEIGHT))
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{emotion}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, emotion, filename)
    
    # Save image
    cv2.imwrite(filepath, face_resized)
    
    return filepath


# ==============================================================================
# MAIN CAPTURE LOOP
# ==============================================================================

def main():
    """
    Main function to run the webcam capture tool.
    """
    
    print("\n" + "="*70)
    print("FACIAL EXPRESSION DATASET COLLECTION TOOL")
    print("="*70)
    print("\nThis tool will help you create your own facial expression dataset!")
    print("\nInstructions:")
    print("1. Position your face in front of the camera")
    print("2. Make the expression you want to capture")
    print("3. Press the number key for that emotion:")
    for key, emotion in EMOTIONS.items():
        print(f"   [{key}] = {emotion.capitalize()}")
    print("4. Try to capture at least 50 images per emotion")
    print("5. Vary your expressions slightly for each capture")
    print("\nTips for better results:")
    print("- Good lighting helps a lot!")
    print("- Keep your face centered")
    print("- Capture from slightly different angles")
    print("- Vary the intensity of your expression")
    print("="*70 + "\n")
    
    input("Press ENTER to start capturing...")
    
    # Setup
    create_emotion_folders()
    counts = count_existing_images()
    
    print("\nCurrent image counts:")
    for emotion, count in counts.items():
        status = "✓" if count >= 50 else "○"
        print(f"  {status} {emotion.capitalize()}: {count} images")
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Initialize camera
    camera_id = 0
    cap = initialize_camera(camera_id)
    
    # State variables
    last_emotion = None
    last_capture_time = 0
    
    print("\n✓ Starting camera feed...")
    print("Press number keys (1-5) to capture, 'Q' to quit\n")
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Detect face
            face_found, face_region, face_coords = detect_face(frame, face_cascade)
            
            # Draw face rectangle if detected
            if face_found:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           FONT, 0.6, (0, 255, 0), 2)
            else:
                # Warning if no face detected
                cv2.putText(frame, "No face detected - position yourself in frame", 
                           (50, 250), FONT, 0.7, (0, 0, 255), 2)
            
            # Draw UI
            display_frame = draw_ui(frame, counts, last_emotion)
            
            # Show frame
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            
            # Switch camera
            elif key == ord('s') or key == ord('S'):
                camera_id = 1 - camera_id  # Toggle between 0 and 1
                cap.release()
                try:
                    cap = initialize_camera(camera_id)
                    print(f"Switched to camera {camera_id}")
                except:
                    print(f"Camera {camera_id} not available, switching back")
                    camera_id = 1 - camera_id
                    cap = initialize_camera(camera_id)
            
            # Capture for emotions (keys 1-5)
            elif chr(key) in EMOTIONS.keys() and face_found:
                import time
                current_time = time.time()
                
                # Check if enough time passed since last capture
                if current_time - last_capture_time >= CAPTURE_DELAY:
                    emotion = EMOTIONS[chr(key)]
                    
                    # Save image
                    filepath = save_image(face_region, emotion)
                    counts[emotion] += 1
                    last_emotion = emotion
                    last_capture_time = current_time
                    
                    print(f"✓ Captured {emotion} (Total: {counts[emotion]})")
                    
                    # Check if reached target
                    if counts[emotion] == 50:
                        print(f"🎉 Great! You've captured 50 {emotion} images!")
            
            # Warning if trying to capture without face
            elif chr(key) in EMOTIONS.keys() and not face_found:
                print("⚠️  No face detected! Position yourself in frame first.")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print("\n" + "="*70)
        print("CAPTURE SESSION COMPLETE")
        print("="*70)
        print("\nFinal image counts:")
        total = 0
        for emotion, count in counts.items():
            status = "✓" if count >= 50 else "○"
            print(f"  {status} {emotion.capitalize()}: {count} images")
            total += count
        
        print(f"\nTotal images captured: {total}")
        
        if all(count >= 50 for count in counts.values()):
            print("\n🎉 Excellent! You have enough images for all emotions!")
            print("You can now train your model by running: python train.py")
        else:
            needed = {e: max(0, 50 - c) for e, c in counts.items()}
            print("\n📝 To reach 50 images per emotion, you still need:")
            for emotion, need in needed.items():
                if need > 0:
                    print(f"   {emotion.capitalize()}: {need} more images")
        
        print("="*70 + "\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point for the webcam capture tool.
    
    Runs when you execute: python webcam_capture.py
    """
    
    try:
        main()
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("- Your webcam is connected")
        print("- You have OpenCV installed: pip install opencv-python")
        print("- No other application is using the webcam")
        import traceback
        traceback.print_exc()


# ==============================================================================
# USAGE NOTES
# ==============================================================================
"""
Installation:
-------------
pip install opencv-python

Usage:
------
python webcam_capture.py

Tips for Collecting Good Data:
-------------------------------
1. Aim for 50-100 images per emotion
2. Vary your expressions slightly
3. Capture from different angles
4. Use good lighting
5. Keep your face centered
6. Don't make exact same expression every time

What to Capture:
----------------
Happy: Smile, laugh, grin
Sad: Frown, downturned mouth
Neutral: Relaxed, no expression
Angry: Furrowed brows, tight lips
Surprised: Raised eyebrows, open mouth

After Capturing:
----------------
Run: python train.py
This will train your model on your captured images!
"""