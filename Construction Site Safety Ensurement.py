import cv2
import numpy as np
import time
import torch
import pygame
import os
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


def setup_environment():
    """Set environment variables for email credentials"""
    os.environ['SAFETY_ALERT_EMAIL'] = 'dharaneeswaran9751sd@gmail.com'
    os.environ['SAFETY_ALERT_PASSWORD'] = 'bhju yrih bsxv bbie'
    os.environ['SAFETY_ALERT_RECIPIENT'] = 'kavas0716@gmail.com'


# Call the function to set environment variables
setup_environment()


class ConstructionSiteSafety:
    def __init__(self):
        self.danger_zones = []
        self.drawing_mode = False
        self.current_points = []
        self.confidence_threshold = 0.6
        self.detection_interval = 200
        self.last_detection_time = 0
        self.model = None
        self.alert_cooldown = 3.0  # seconds between alerts
        self.last_alert_time = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize video playback control variables
        self.playback_speed = 1.0  # Normal speed
        self.frame_skip = 0  # No frames skipped initially
        self.max_speed = 5.0
        self.min_speed = 0.2


        self.latest_frame = None
        self.latest_results = None
        self.detection_thread = None
        self.detection_running = False


        pygame.mixer.init()

        self.sounds_dir = r"C:\Users\vishw\PyCharmMiscProject\og scripts\PythonProject\Script files\construction site\sound cons"
        os.makedirs(self.sounds_dir, exist_ok=True)


        self.alert_sounds = {
            'danger': self.load_sound('danger_alert.wav'),
            'warning': self.load_sound('warning.wav'),
            'emergency': self.load_sound('emergency.wav')
        }


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

    def load_sound(self, filename):
        """Load a sound file, return None if file not found"""
        try:
            sound_path = os.path.join(self.sounds_dir, filename)
            if os.path.exists(sound_path):
                return pygame.mixer.Sound(sound_path)
            else:
                print(f"Warning: Sound file not found at {sound_path}")
                return None
        except Exception as e:
            print(f"Warning: Could not load sound file {filename}: {str(e)}")
            return None

    def load_model(self):
        """Load the YOLOv5 detection model"""
        try:
            print("Loading detection model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        return True

    def start_detection_thread(self):
        """Start a separate thread for object detection"""
        if self.detection_thread is not None and self.detection_thread.is_alive():
            return  # Thread already running

        self.detection_running = True
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def detection_loop(self):
        """Continuous detection loop running in a separate thread"""
        while self.detection_running:
            if self.latest_frame is not None:
                try:

                    frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)


                    height, width = frame_rgb.shape[:2]
                    scale_factor = 0.5
                    resized_frame = cv2.resize(frame_rgb, (int(width * scale_factor), int(height * scale_factor)))


                    self.latest_results = self.model(resized_frame)


                    time_to_sleep = max(0, self.detection_interval / 1000)
                    time.sleep(time_to_sleep)
                except Exception as e:
                    print(f"Error in detection thread: {str(e)}")
                    time.sleep(1)
        print("Detection thread stopped")

    def send_email_alert(self, image_path, people_count):
        """Send email alert with image attachment"""
        current_time = time.time()
        if current_time - self.last_email_time < self.email_cooldown:
            return

        try:
            msg = MIMEMultipart()
            msg['Subject'] = 'SAFETY ALERT: People Detected in Danger Zone'
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']

            text = f'''SAFETY ALERT!

Number of people detected in danger zone: {people_count}
Time of detection: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is an automated alert from the Construction Site Safety System.'''

            msg.attach(MIMEText(text))

            try:
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(img)
            except Exception as e:
                print(f"Warning: Could not attach image: {str(e)}")

            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)

            self.last_email_time = current_time
            print(f"Email alert sent to {self.email_config['recipient_email']}")

        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")

    def close_browser_tab(self):
        """Function to ensure browser tabs are closed"""
        try:
            time.sleep(7)  # Wait longer before attempting to close
            if platform.system() == "Windows":
                import pyautogui
                pyautogui.hotkey('ctrl', 'w')  # Close tab with keyboard shortcut
        except Exception as e:
            print(f"Failed to close tab: {str(e)}")

    def kill_browser_processes(self):
        """Kill any hanging browser processes"""
        try:
            if platform.system() == "Windows":
                os.system("taskkill /f /im chrome.exe")
            elif platform.system() == "Linux":
                os.system("pkill chrome")
            elif platform.system() == "Darwin":  # macOS
                os.system("pkill -f Chrome")
        except Exception as e:
            print(f"Failed to kill browser processes: {str(e)}")

    def send_whatsapp_alert(self, people_count, image_path=None):
        """Send WhatsApp alert about people in danger zones"""
        current_time = time.time()
        if current_time - self.last_whatsapp_time < self.whatsapp_config['cooldown']:
            return False

        if not self.whatsapp_config['enabled'] or not self.whatsapp_config['recipients']:
            print("WhatsApp alerts are disabled or no recipients configured")
            return False

        try:

            alert_message = f"""🚨 SAFETY ALERT 🚨

{people_count} {'person' if people_count == 1 else 'people'} detected in danger zone!
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from the Construction Site Safety System."""

            # Get current time (add 2 minutes to ensure we're sending in the future)
            now = datetime.now()
            future_time = now + timedelta(minutes=2)

            success_count = 0

            for phone_number in self.whatsapp_config['recipients']:
                try:

                    phone = phone_number.strip()
                    if not phone.startswith('+'):
                        print(f"Warning: Phone number {phone} doesn't have country code")
                        continue

                    print(f"Attempting to send WhatsApp message to {phone}")

                    # Calculate minutes for this recipient (staggered)
                    recipient_minute = future_time.minute + (0 if success_count == 0 else 1)
                    if recipient_minute >= 60:
                        hour_offset = recipient_minute // 60
                        recipient_minute = recipient_minute % 60
                        recipient_hour = future_time.hour + hour_offset
                    else:
                        recipient_hour = future_time.hour


                    pywhatkit.sendwhatmsg(
                        phone,
                        alert_message,
                        recipient_hour,
                        recipient_minute,
                        wait_time=45,  # Increased wait time
                        tab_close=True,
                        close_time=15  # Increased close time
                    )


                    threading.Thread(target=self.close_browser_tab, daemon=True).start()

                    success_count += 1
                    print(f"WhatsApp alert scheduled to {phone}")


                    time.sleep(8)


                    if image_path and os.path.exists(image_path):
                        try:
                            pywhatkit.sendwhats_image(
                                phone,
                                image_path,
                                "Safety Alert: Danger zone detection image",
                                wait_time=45,
                                tab_close=True,
                                close_time=15
                            )


                            threading.Thread(target=self.close_browser_tab, daemon=True).start()

                            print(f"WhatsApp image alert scheduled to {phone}")
                        except Exception as e:
                            print(f"Failed to schedule WhatsApp image to {phone}: {str(e)}")

                except Exception as e:
                    print(f"Failed to schedule WhatsApp to {phone}: {str(e)}")

            if success_count > 0:
                self.last_whatsapp_time = current_time
                return True
            return False

        except Exception as e:
            print(f"WhatsApp alert failed: {str(e)}")
            return False

    def play_alert(self, alert_type='danger'):
        """Play alert sound in a separate thread"""

        def play_sound():
            if alert_type in self.alert_sounds and self.alert_sounds[alert_type]:
                self.alert_sounds[alert_type].play()
            else:
                frequency = 2500
                duration = 1000
                winsound.Beep(frequency, duration)

        threading.Thread(target=play_sound, daemon=True).start()

    def add_danger_zone(self, points):
        """Add a danger zone defined by points"""
        if len(points) >= 3:
            self.danger_zones.append(np.array(points, dtype=np.int32))

    def is_point_in_any_danger_zone(self, point):
        """Check if a point is in any danger zone"""
        for zone in self.danger_zones:
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False

    def draw_danger_zones(self, frame):
        """Draw all danger zones on the frame"""
        overlay = frame.copy()

        for zone in self.danger_zones:
            cv2.fillPoly(overlay, [zone], (0, 0, 255))

        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for zone in self.danger_zones:
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

        if self.drawing_mode and self.current_points:
            for point in self.current_points:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

        return frame

    def process_detections(self, frame, detections):
        """Process and draw detections"""
        height, width, _ = frame.shape
        people_in_danger = 0
        total_people = 0

        if detections.pred is not None and len(detections.pred[0]) > 0:
            for det in detections.pred[0]:
                if det[-1] == 0:  # YOLOv5 class 0 is person
                    confidence = det[4].item()
                    if confidence > self.confidence_threshold:
                        total_people += 1
                        xmin, ymin, xmax, ymax = map(int, det[:4].cpu().numpy())

                        # Check if person is in danger zone
                        person_position = (int((xmin + xmax) / 2), ymax)
                        in_danger = self.is_point_in_any_danger_zone(person_position)

                        if in_danger:
                            people_in_danger += 1
                            color = (0, 0, 255)  # Red for danger
                        else:
                            color = (0, 255, 0)  # Green for safe

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        label = f'Person {"DANGER" if in_danger else "SAFE"} {int(confidence * 100)}%'
                        cv2.putText(frame, label, (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, total_people, people_in_danger

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing danger zones"""
        if self.drawing_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.current_points) >= 3:
                    self.add_danger_zone(self.current_points)
                self.current_points = []

    def process_video(self, source):
        """Process video with danger zone and human detection"""
        if isinstance(source, str) and not os.path.exists(source):
            print(f"Error: Video file not found at {source}")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return


        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {original_fps}")


        target_fps = original_fps
        frame_delay = int(1000 / target_fps)

        window_name = 'Construction Site Safety'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        if not self.load_model():
            return


        self.start_detection_thread()

        print("/nControls:")
        print("- Press 'd' to enter/exit drawing mode")
        print("- In drawing mode:")
        print("  * Left click to add points")
        print("  * Right click to finish current zone")
        print("- Press 'c' to clear all zones")
        print("- Press 'r' to reset playback to normal speed")
        print("- Press 'q' to quit")


        results = None
        last_frame_time = time.time()

        while True:
            # Skip frames based on frame_skip value
            for _ in range(self.frame_skip):
                ret, _ = cap.read()  # Read and discard frames
                if not ret:
                    break

            # Read the frame we'll actually process
            ret, frame = cap.read()
            if not ret:
                break

            # Store frame for detection thread
            self.latest_frame = frame.copy()

            # Use latest results from detection thread
            if self.latest_results is not None:
                results = self.latest_results

            current_time = time.time()

            # Draw danger zones
            frame = self.draw_danger_zones(frame)

            # Process detections only if we have results
            if results is not None:
                frame, total_people, people_in_danger = self.process_detections(frame, results)

                # Show status and counts
                cv2.putText(frame, f"Drawing Mode: {'ON' if self.drawing_mode else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Total People: {total_people}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"People in Danger: {people_in_danger}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Speed: {self.playback_speed:.1f}x | Skip: {self.frame_skip}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Handle alerts
                if people_in_danger > 0 and (current_time - self.last_alert_time) >= self.alert_cooldown:
                    if people_in_danger >= 3:
                        self.play_alert('emergency')
                    elif people_in_danger == 2:
                        self.play_alert('warning')
                    else:
                        self.play_alert('danger')

                    self.last_alert_time = current_time

                    # Save frame and send alerts
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"alert_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Alert! {people_in_danger} people in danger zones. Image saved as {filename}")

                    # Send email alert in background thread
                    threading.Thread(target=self.send_email_alert, args=(filename, people_in_danger),
                                     daemon=True).start()

                    # Send WhatsApp alert if enabled
                    if self.whatsapp_config['enabled']:
                        threading.Thread(target=self.send_whatsapp_alert, args=(people_in_danger, filename),
                                         daemon=True).start()
            else:
                # If no detection results yet, show loading message
                cv2.putText(frame, "Loading detection model...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)

            # Control frame rate with adjusted delay based on playback speed
            adjusted_delay = max(1, int(frame_delay / self.playback_speed))
            key = cv2.waitKey(adjusted_delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):
                self.drawing_mode = not self.drawing_mode
                if not self.drawing_mode and self.current_points:
                    if len(self.current_points) >= 3:
                        self.add_danger_zone(self.current_points)
                    self.current_points = []
            elif key == ord('c'):
                self.danger_zones = []
                self.current_points = []
            elif key == ord('+') or key == ord('='):  # Both keys for convenience
                self.playback_speed = min(self.max_speed, self.playback_speed + 0.1)
                print(f"Playback speed: {self.playback_speed:.1f}x")
            elif key == ord('-'):
                self.playback_speed = max(self.min_speed, self.playback_speed - 0.1)
                print(f"Playback speed: {self.playback_speed:.1f}x")
            elif key == ord(']'):  # Increase frame skipping (faster playback)
                self.frame_skip = min(10, self.frame_skip + 1)
                print(f"Frame skip: {self.frame_skip}")
            elif key == ord('['):  # Decrease frame skipping (slower playback)
                self.frame_skip = max(0, self.frame_skip - 1)
                print(f"Frame skip: {self.frame_skip}")
            elif key == ord('r'):  # Reset to normal speed
                self.playback_speed = 1.0
                self.frame_skip = 0
                print("Playback reset to normal speed")

        # Stop detection thread when exiting
        self.detection_running = False
        if self.detection_thread is not None:
            self.detection_thread.join(timeout=1.0)

        cap.release()
        cv2.destroyAllWindows()


def check_camera(source=0):
    """Check if camera is available"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return False
    cap.release()
    return True


if __name__ == "__main__":
    safety_system = ConstructionSiteSafety()

    # Configure WhatsApp recipients
    safety_system.whatsapp_config['recipients'] = [
        '+917200811012',  # This is correctly formatted
      
    ]

    path = r"C:\Users\vishw\PyCharmMiscProject\og scripts\PythonProject\Script files\construction site\WhatsApp Video 2025-05-02 at 15.44.03_69b23286.mp4"

    if os.path.exists(video_path):
        print(f"Using video file: {video_path}")
        safety_system.process_video(video_path)
    elif check_camera(0):
        print("Using webcam...")
        safety_system.process_video(0)
    else:
        print("Error: Neither video file nor camera is available")
