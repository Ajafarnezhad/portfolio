import argparse
import logging
import sys
import time
import cv2
import pyautogui as robot
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, boundary_left: int = 400, boundary_right: int = 900, boundary_top: int = 100, boundary_bottom: int = 600,
                 scale_factor: float = 1.1, min_neighbors: int = 5, smoothing_factor: float = 0.5):
        """Initialize the FaceTracker with configurable parameters."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.boundary_top = boundary_top
        self.boundary_bottom = boundary_bottom
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.smoothing_factor = smoothing_factor  # For smoothing mouse movement
        self.prev_mouse_x = robot.position().x
        self.prev_mouse_y = robot.position().y
        self.cap = None

    def start_camera(self, camera_index: int = 0) -> None:
        """Start the camera feed."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error("Could not open camera.")
            sys.exit(1)
        logger.info("Camera started.")

    def track_face(self, show_video: bool = True) -> None:
        """Track face and eyes, control mouse based on face position."""
        if self.cap is None:
            self.start_camera()
        
        logger.info("Starting face tracking. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors)
            
            if len(faces) > 0:
                # Take the largest face (assuming closest)
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces[0]
                x2, y2 = x + w, y + h
                gray_face = gray[y:y2, x:x2]
                
                # Detect eyes in face region
                eyes = self.eye_cascade.detectMultiScale(gray_face, self.scale_factor, self.min_neighbors, minSize=(30, 30))
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
                
                # Draw eyes (up to 2)
                eye_count = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                    eye_count += 1
                    if eye_count == 2:
                        break
                
                # Mouse control with smoothing
                self._control_mouse(x, y, x2, y2)
            
            # Draw boundary rectangle
            color = (255, 255, 255) if len(faces) == 0 else (0, 0, 255)  # White if no face, red if moving
            cv2.rectangle(frame, (self.boundary_left, self.boundary_top), (self.boundary_right, self.boundary_bottom), color, 2)
            
            if show_video:
                cv2.imshow('Face Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Face tracking stopped.")
        robot.moveTo(0, 0, duration=0.5)  # Reset mouse to corner

    def _control_mouse(self, x: int, y: int, x2: int, y2: int) -> None:
        """Control mouse based on face position with smoothing."""
        current_mouse_x, current_mouse_y = robot.position()
        
        delta_x = 0
        delta_y = 0
        
        if x < self.boundary_left:
            delta_x = -(self.boundary_left - x)
        elif x2 > self.boundary_right:
            delta_x = x2 - self.boundary_right
        
        if y < self.boundary_top:
            delta_y = -(self.boundary_top - y)
        elif y2 > self.boundary_bottom:
            delta_y = y2 - self.boundary_bottom
        
        # Apply smoothing
        new_mouse_x = int(self.prev_mouse_x + self.smoothing_factor * delta_x)
        new_mouse_y = int(self.prev_mouse_y + self.smoothing_factor * delta_y)
        
        robot.moveTo(new_mouse_x, new_mouse_y, duration=0)
        
        self.prev_mouse_x = new_mouse_x
        self.prev_mouse_y = new_mouse_y
        
        if delta_x != 0 or delta_y != 0:
            logger.debug(f"Mouse moved to ({new_mouse_x}, {new_mouse_y})")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Face Tracker for Mouse Control using OpenCV.")
    parser.add_argument("--boundary_left", type=int, default=400, help="Left boundary for face detection.")
    parser.add_argument("--boundary_right", type=int, default=900, help="Right boundary for face detection.")
    parser.add_argument("--boundary_top", type=int, default=100, help="Top boundary for face detection.")
    parser.add_argument("--boundary_bottom", type=int, default=600, help="Bottom boundary for face detection.")
    parser.add_argument("--scale_factor", type=float, default=1.1, help="Scale factor for cascade detection.")
    parser.add_argument("--min_neighbors", type=int, default=5, help="Min neighbors for cascade detection.")
    parser.add_argument("--smoothing_factor", type=float, default=0.5, help="Smoothing factor for mouse movement (0-1).")
    parser.add_argument("--no_video", action="store_true", help="Run without displaying video feed.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    tracker = FaceTracker(
        boundary_left=args.boundary_left,
        boundary_right=args.boundary_right,
        boundary_top=args.boundary_top,
        boundary_bottom=args.boundary_bottom,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        smoothing_factor=args.smoothing_factor
    )
    tracker.track_face(show_video=not args.no_video)