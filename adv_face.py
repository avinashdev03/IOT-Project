import cv2
import os
import smtplib
import ssl
import threading
import datetime
import numpy as np
from deepface import DeepFace  # <-- NEW LIBRARY

# --- 1. Email Alert Configuration (NEEDS SETUP) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.environ.get('MY_EMAIL')
SENDER_PASSWORD = os.environ.get('MY_EMAIL_PASSWORD')

# --- 2. Name-to-Email Database (NEEDS SETUP) ---
EMAIL_MAP = {
    "sanjay": "sanjay@example.com",
    # Add all your known faces and their emails here
}

# --- 3. One-Time Alert System ---
sent_alert_list = []

# --- 4. Database and Model Setup ---
DB_PATH = "known_faces"  # Path to your folder of known faces
MODEL_NAME = "VGG-Face"  # Use this model for high accuracy
DETECTOR_BACKEND = "opencv"  # A fast detector
# Build the model once (this will be slow the first time)
print("Loading face recognition model...")
DeepFace.find(
    img_path=np.zeros((100, 100, 3), dtype=np.uint8),
    db_path=DB_PATH,
    model_name=MODEL_NAME,
    detector_backend=DETECTOR_BACKEND
)
print("Model loaded. Ready to connect to camera.")


# --- 5. Email Sending Function ---
# (This function is the same as before)
def send_alert_email(recipient_name, recipient_email):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Error: Email credentials not set in environment variables.")
        return
    context = ssl.create_default_context()
    subject = "Security Alert: Face Detected"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = f"Hello {recipient_name.capitalize()},\n\nYour face was detected by the system at {now}."
    message = f"Subject: {subject}\n\n{body}"
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message)
            print(f"--- Alert email successfully sent to {recipient_name} ---")
    except Exception as e:
        print(f"Error sending email: {e}")


# --- 6. Initialize CCTV Stream ---
rtsp_url = "rtsp://YOUR_USERNAME:YOUR_PASSWORD@YOUR_CAMERA_IP/stream_path"
# rtsp_url = 0  # Use 0 for webcam
video_capture = cv2.VideoCapture(rtsp_url)

if not video_capture.isOpened():
    print("Error: Could not open video stream. Check RTSP URL.")
    exit()

print("Connecting to camera feed...")

# --- 7. Main Real-Time Loop ---
frame_count = 0
# ⬇ ⬇ ⬇ WE MUST SKIP MORE FRAMES. This will be slow. ⬇ ⬇ ⬇
process_every_n_frames = 15  # Process 1 out of 15 frames

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Lost connection to camera feed.")
        break

    frame_count += 1

    # ⬇ ⬇ ⬇ NEW FRAME SKIPPING LOGIC ⬇ ⬇ ⬇
    if frame_count % process_every_n_frames == 0:

        try:
            # --- 8. Run High-Accuracy Recognition ---
            # This is the new function. It finds and recognizes all faces
            # It compares the 'frame' against all images in 'DB_PATH'
            dfs = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,  # Don't re-detect faces if not needed
                silent=True  # Suppress console logs for each find
            )

            # 'dfs' is a list of dataframes. We need to process it.
            if dfs and not dfs[0].empty:
                for index, row in dfs[0].iterrows():
                    # Get the name (filename) of the matched person
                    # The path will be like 'known_faces/sanjay.jpg'
                    identity_path = row['identity']
                    name = os.path.splitext(os.path.basename(identity_path))[0]

                    # Get the bounding box
                    x = row['source_x']
                    y = row['source_y']
                    w = row['source_w']
                    h = row['source_h']

                    # Draw the box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    # --- Check for Alert ---
                    if name != "Unknown" and name in EMAIL_MAP:
                        if name not in sent_alert_list:
                            print(f"Recognized {name} for the first time. Sending alert...")
                            sent_alert_list.append(name)
                            recipient_email = EMAIL_MAP[name]
                            email_thread = threading.Thread(
                                target=send_alert_email,
                                args=(name, recipient_email)
                            )
                            email_thread.start()

        except Exception as e:
            # This can happen if no faces are in the frame
            # print(f"No face detected or error: {e}")
            pass  # Silently continue

    # --- 9. Display the Result (happens every frame) ---
    display_frame = cv2.resize(frame, (1280, 720))  # Resize to a 720p window
    cv2.imshow('CCTV Feed (High Accuracy)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. Cleanup ---
video_capture.release()
cv2.destroyAllWindows()