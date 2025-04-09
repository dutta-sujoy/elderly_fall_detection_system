import os
import time
from datetime import datetime

def send_alert(image_path=None):
    """
    Send alert when a fall is detected
    
    In a real implementation, this could:
    - Send SMS using Twilio
    - Send email
    - Trigger a local alarm
    - Call emergency services
    
    This is a simplified version that logs the alert
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Print alert to console
    print("\n" + "="*50)
    print(f"⚠️ FALL DETECTED at {timestamp} ⚠️")
    print(f"Image saved: {image_path if image_path else 'No image'}")
    print("="*50 + "\n")
    
    # In a real system, you would add code here to:
    # 1. Send SMS alert
    # 2. Send email with the image attached
    # 3. Trigger a local alarm sound
    
    # Log the alert to a file
    with open("fall_alerts.log", "a") as log_file:
        log_file.write(f"{timestamp} - Fall detected. Image: {image_path}\n")
    
    # Simulate alert processing time
    time.sleep(1)
    
    return True
