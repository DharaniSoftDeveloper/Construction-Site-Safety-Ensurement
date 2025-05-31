
import numpy as np
import time
import torch
import pygame
import os
import loggin
from datetime import datetime, timedelta
import threading
import winsound
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
import pywhatkit
import random
import platform
import yam
from pathlib import Path


# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"safety_system_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config():
    """Load configuration from YAML file"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def setup_environment():
    """Set environment variables for email credentials"""
    os.environ['SAFETY_ALERT_EMAIL'] = 'dharaneeswaran9751sd@gmail.com'
    os.environ['SAFETY_ALERT_PASSWORD'] = 'bhju yrih bsxv bbie'
    os.environ['SAFETY_ALERT_RECIPIENT'] = 'kavas0716@gmail.com'


# Initialize logging and load config
setup_logging()
config = load_config()

# Call the function to set environment variables
setup_environment()


class ConstructionSiteSafety:
    def __init__(self):
        self.danger_zones = []
        self.drawing_mode = False
        self.current_points = []
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.detection_interval = config.get('detection_interval', 200)
        self.last_detection_time = 0
        self.model = None
        self.alert_cooldown = config.get('alert_cooldown', 3.0)
        self.last_alert_time = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Enhanced detection settings
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.model_size = config.get('model_size', 's')  # Can be 's', 'm', 'l', or 'x'
        self.min_detection_size = config.get('min_detection_size', 30)
        self.tracking_enabled = config.get('tracking_enabled', True)

        # Initialize video playback control variables
        self.playback_speed = 1.0
        self.frame_skip = 0
        self.max_speed = 5.0
        self.min_speed = 0.2

        self.latest_frame = None
        self.latest_results = None
        self.detection_thread = None
        self.detection_running = False

        # Enhanced alert types
        self.alert_levels = {
            'low': {'color': (255, 255, 0), 'sound': 'warning.wav'},
            'medium': {'color': (0, 165, 255), 'sound': 'danger_alert.wav'},
            'high': {'color': (0, 0, 255), 'sound': 'emergency.wav'},
            'critical': {'color': (0, 0, 255), 'sound': 'critical.wav'}
        }

        pygame.mixer.init()

        self.sounds_dir = Path("sounds")
        self.sounds_dir.mkdir(exist_ok=True)

        self.alert_sounds = self._load_alert_sounds()

        self.email_config = {
            'sender_email': os.getenv('SAFETY_ALERT_EMAIL'),
            'sender_password': os.getenv('SAFETY_ALERT_PASSWORD'),
            'recipient_email': os.getenv('SAFETY_ALERT_RECIPIENT'),
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
        }
        self.last_email_time = 0
        self.email_cooldown = 300

        self.whatsapp_config = {
            'enabled': False,
            'recipients': [],
            'cooldown': 300
        }
        self.last_whatsapp_time = 0

        logging.info("Construction Site Safety System initialized")

    def _load_alert_sounds(self):
        """Load all alert sounds"""
        sounds = {}
        for level, config in self.alert_levels.items():
            sound_file = self.sounds_dir / config['sound']
            try:
                if sound_file.exists():
                    sounds[level] = pygame.mixer.Sound(str(sound_file))
                    logging.info(f"Loaded sound for alert level: {level}")
                else:
                    logging.warning(f"Sound file not found: {sound_file}")
            except Exception as e:
                logging.error(f"Failed to load sound for {level}: {e}")
        return sounds

    def load_model(self):
        """Load the YOLOv5 detection model with enhanced settings"""
        try:
            logging.info(f"Loading YOLOv5{self.model_size} model...")
            self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{self.model_size}', pretrained=True)
            self.model.to(self.device)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            self.model.eval()
            logging.info("Model loaded successfully!")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def determine_alert_level(self, people_in_danger, total_people):
        """Determine alert level based on situation severity"""
        danger_ratio = people_in_danger / total_people if total_people > 0 else 0
        
        if people_in_danger >= 5 or danger_ratio >= 0.8:
            return 'critical'
        elif people_in_danger >= 3 or danger_ratio >= 0.5:
            return 'high'
        elif people_in_danger >= 2:
            return 'medium'
        elif people_in_danger >= 1:
            return 'low'
        return None

    def process_detections(self, frame, detections):
        """Process and draw detections with enhanced visualization"""
        height, width, _ = frame.shape
        people_in_danger = 0
        total_people = 0
        detected_people = []

        if detections.pred is not None and len(detections.pred[0]) > 0:
            for det in detections.pred[0]:
                if det[-1] == 0:  # YOLOv5 class 0 is person
                    confidence = det[4].item()
                    if confidence > self.confidence_threshold:
                        xmin, ymin, xmax, ymax = map(int, det[:4].cpu().numpy())
                        
                        # Filter out small detections
                        if (xmax - xmin) < self.min_detection_size or (ymax - ymin) < self.min_detection_size:
                            continue

                        total_people += 1
                        person_position = (int((xmin + xmax) / 2), ymax)
                        in_danger = self.is_point_in_any_danger_zone(person_position)

                        detected_people.append({
                            'bbox': (xmin, ymin, xmax, ymax),
                            'confidence': confidence,
                            'in_danger': in_danger,
                            'position': person_position
                        })

                        if in_danger:
                            people_in_danger += 1

        # Draw detections with enhanced visualization
        for person in detected_people:
            xmin, ymin, xmax, ymax = person['bbox']
            color = (0, 0, 255) if person['in_danger'] else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw danger zone indicator
            if person['in_danger']:
                cv2.circle(frame, person['position'], 5, (0, 0, 255), -1)
                cv2.line(frame, person['position'], 
                        (person['position'][0], person['position'][1] - 20),
                        (0, 0, 255), 2)
            
            # Add label with confidence
            label = f"{'DANGER' if person['in_danger'] else 'SAFE'} {int(person['confidence'] * 100)}%"
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, total_people, people_in_danger

    def process_video(self, source):
        """Process video with enhanced monitoring and alerts"""
        if isinstance(source, str) and not os.path.exists(source):
            logging.error(f"Video file not found: {source}")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error("Could not open video source")
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Original video FPS: {original_fps}")

        target_fps = original_fps
        frame_delay = int(1000 / target_fps)

        window_name = 'Construction Site Safety'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        if not self.load_model():
            return

        self.start_detection_thread()

        logging.info("Starting video processing...")
        
        while True:
            for _ in range(self.frame_skip):
                ret, _ = cap.read()
                if not ret:
                    break

            ret, frame = cap.read()
            if not ret:
                break

            self.latest_frame = frame.copy()

            if self.latest_results is not None:
                results = self.latest_results

                frame = self.draw_danger_zones(frame)

                if results is not None:
                    frame, total_people, people_in_danger = self.process_detections(frame, results)
                    
                    # Enhanced status display
                    self.draw_status_overlay(frame, total_people, people_in_danger)
                    
                    # Handle alerts based on severity
                    alert_level = self.determine_alert_level(people_in_danger, total_people)
                    if alert_level and time.time() - self.last_alert_time >= self.alert_cooldown:
                        self.trigger_alerts(alert_level, people_in_danger, frame)

            cv2.imshow(window_name, frame)

            if self.handle_keyboard_input(cv2.waitKey(max(1, int(frame_delay / self.playback_speed)))):
                break

        self.cleanup()
        logging.info("Video processing ended")

    def draw_status_overlay(self, frame, total_people, people_in_danger):
        """Draw enhanced status overlay with additional information"""
        overlay = frame.copy()
        overlay_height = 150
        cv2.rectangle(overlay, (0, 0), (400, overlay_height), (0, 0, 0), -1)
        
        # Add status information with improved formatting
        status_text = [
            f"Drawing Mode: {'ON' if self.drawing_mode else 'OFF'}",
            f"Total People: {total_people}",
            f"People in Danger: {people_in_danger}",
            f"Speed: {self.playback_speed:.1f}x | Skip: {self.frame_skip}",
            f"FPS: {int(1.0 / (time.time() - self.last_detection_time) if self.last_detection_time else 0)}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(overlay, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def trigger_alerts(self, alert_level, people_in_danger, frame):
        """Trigger appropriate alerts based on severity level"""
        logging.info(f"Triggering {alert_level} alert for {people_in_danger} people in danger")
        
        # Play alert sound
        if alert_level in self.alert_sounds:
            self.alert_sounds[alert_level].play()
        
        # Save incident image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{alert_level}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        
        # Send notifications
        alert_message = f"SAFETY ALERT ({alert_level.upper()}): {people_in_danger} people in danger zones"
        
        threading.Thread(target=self.send_email_alert,
                        args=(filename, people_in_danger, alert_level),
                        daemon=True).start()
        
        if self.whatsapp_config['enabled']:
            threading.Thread(target=self.send_whatsapp_alert,
                            args=(people_in_danger, filename, alert_level),
                            daemon=True).start()
        
        self.last_alert_time = time.time()

    def cleanup(self):
        """Cleanup resources"""
        self.detection_running = False
        if self.detection_thread is not None:
            self.detection_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        logging.info("Cleanup completed")

    def handle_keyboard_input(self, key):
        """Handle keyboard inputs"""
        if key == ord('q'):
            return True
        elif key == ord('d'):
            self.drawing_mode = not self.drawing_mode
            if not self.drawing_mode and self.current_points:
                if len(self.current_points) >= 3:
                    self.add_danger_zone(self.current_points)
                self.current_points = []
        elif key == ord('c'):
            self.danger_zones = []
            self.current_points = []
        elif key in [ord('+'), ord('=')]:
            self.playback_speed = min(self.max_speed, self.playback_speed + 0.1)
        elif key == ord('-'):
            self.playback_speed = max(self.min_speed, self.playback_speed - 0.1)
        elif key == ord(']'):
            self.frame_skip = min(10, self.frame_skip + 1)
        elif key == ord('['):
            self.frame_skip = max(0, self.frame_skip - 1)
        elif key == ord('r'):
            self.playback_speed = 1.0
            self.frame_skip = 0
        return False


def check_camera(source=0):
    """Check if camera is available"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Camera not available")
        return False
    cap.release()
    return True


if __name__ == "__main__":
    safety_system = ConstructionSiteSafety()

    # Configure WhatsApp recipients
    safety_system.whatsapp_config['recipients'] = [
        '+917200811012',
    ]

    video_path = r"C:\Users\vishw\PyCharmMiscProject\og scripts\PythonProject\Script files\construction site\WhatsApp Video 2025-05-02 at 15.44.03_69b23286.mp4"

    if os.path.exists(video_path):
        logging.info(f"Using video file: {video_path}")
        safety_system.process_video(video_path)
    elif check_camera(0):
        logging.info("Using webcam...")
        safety_system.process_video(0)
    else:
        logging.error("Neither video file nor camera is available")