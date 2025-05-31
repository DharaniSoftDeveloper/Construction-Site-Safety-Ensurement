# Construction Site Safety Ensurement

An intelligent video monitoring system that uses computer vision and machine learning to ensure safety compliance in construction sites.

## 🚀 Features

- **Real-time Person Detection**: Uses YOLOv5 to detect people in video feeds
- **Danger Zone Definition**: Interactive drawing of danger zones on the video feed
- **Alert System**: Multi-channel alerts through:
  - Visual indicators
  - Audio alerts (multiple sound types)
  - Email notifications
  - WhatsApp messages
- **Playback Controls**: Variable speed playback and frame skipping
- **Real-time Statistics**: Display of total people and people in danger zones
- **Image Capture**: Automatic saving of incident images

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Pygame
- PyWhatKit
- CUDA-capable GPU (optional, for faster detection)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Construction-Site-Safety-Ensurement.git
cd Construction-Site-Safety-Ensurement
```

2. Install required packages:
```bash
pip install torch torchvision opencv-python pygame pywhatkit
```

3. Configure environment variables for email notifications:
```python
SAFETY_ALERT_EMAIL=your-email@gmail.com
SAFETY_ALERT_PASSWORD=your-app-specific-password
SAFETY_ALERT_RECIPIENT=recipient@email.com
```

## 🎮 Usage

1. Run the main script:
```bash
python "Construction Site Safety Ensurement.py"
```

2. Controls:
- `d` - Enter/exit drawing mode
- Left click - Add points to danger zone (in drawing mode)
- Right click - Complete current danger zone (in drawing mode)
- `c` - Clear all danger zones
- `+`/`-` - Adjust playback speed
- `[`/`]` - Adjust frame skip
- `r` - Reset playback to normal speed
- `q` - Quit application

## ⚙️ Configuration

### WhatsApp Alerts
Add recipient phone numbers in the main script:
```python
safety_system.whatsapp_config['recipients'] = [
    '+1234567890',  # Include country code
]
```

### Email Alerts
Configure email settings in the environment variables or directly in the script:
```python
email_config = {
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-specific-password',
    'recipient_email': 'recipient@email.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}
```

## 🔒 Security Notes

- Use app-specific passwords for email authentication
- Keep environment variables secure
- Ensure proper access controls for the video feed
- Regularly update dependencies for security patches

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.