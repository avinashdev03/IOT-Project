import cv2
import os

# --- 1. Setup ---
# The folder where we will store the known faces
output_folder = "known_faces"

# Create the folder if it doesn't already exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# --- 2. Get User's Name ---
# Ask for the name in the terminal
# This name will be the filename (e.g., "sanjay")
name = input("Enter the person's name (no spaces): ")
if " " in name:
    print("Error: Name should not contain spaces. Please restart.")
    exit()

# Final filename, e.g., "known_faces/sanjay.jpg"
file_name = os.path.join(output_folder, f"{name}.jpg")

# --- 3. Initialize Webcam ---
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam opened. Look at the camera.")
print("Press 's' to save your photo and quit.")
print("Press 'q' to quit without saving.")

while True:
    # --- 4. Read Frame from Webcam ---
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display a smaller window for convenience
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Draw a simple box in the middle to help with centering
    h, w, _ = small_frame.shape
    center_x, center_y = w // 2, h // 2
    box_size = 150

    cv2.rectangle(
        small_frame,
        (center_x - box_size, center_y - box_size),
        (center_x + box_size, center_y + box_size),
        (0, 255, 0),
        2
    )

    # Display the resulting image
    cv2.imshow('Press "s" to save, "q" to quit', small_frame)

    # --- 5. Wait for Key Press ---
    key = cv2.waitKey(1) & 0xFF

    # Save photo on 's' key
    if key == ord('s'):
        # Save the original high-resolution frame, not the small one
        cv2.imwrite(file_name, frame)
        print(f"\nSuccess! Image saved as: {file_name}")
        break  # Exit the loop after saving

    # Quit on 'q' key
    elif key == ord('q'):
        print("\nQuit without saving.")
        break  # Exit the loop

# --- 6. Release Handle to the Webcam ---
video_capture.release()
cv2.destroyAllWindows()