import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/pinbowling.pt")

cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("Failed to open /dev/video0, trying /dev/video1...")
    cap = cv2.VideoCapture("/dev/video1")

if not cap.isOpened():
    raise ValueError("Error: Could not open any video capture device.")

bbox_fixed = None
frame_count = 0
fix_frame = 2
no_detection_count = 0
max_no_detection = 5
last_frame = None
height_threshold_ratio = 0.25  # heigth threshold
strike = 0

def count_standing_pins(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 3), 0)

    edges = cv2.Canny(blur, 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # left, middle, right
    width = roi.shape[1]
    height = roi.shape[0]
    height_threshold = int(height * height_threshold_ratio)  # heigth gain
    left_end = width // 3
    right_start = 2 * (width // 3)

    left_count = 0
    middle_count = 0
    right_count = 0
    global strike
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 0 < area < 1500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if area >= 1  and y + h >= height_threshold and aspect_ratio >= 0.20: # normal found pin stand is 0.2 <= a <= 1.5
                print(f"y+h = {aspect_ratio}")
                pin_x = x + w // 2
                cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
                cv2.line(roi, (0, height_threshold), (width, height_threshold), (255, 0, 0), 2)

                if pin_x < left_end:
                    left_count += 1
                elif left_end <= pin_x < right_start:
                    middle_count += 1
                else:
                    right_count += 1

                cv2.line(roi, (pin_x, 0), (pin_x, roi.shape[0]), (0, 0, 255), 2)

    if left_count > 0 or middle_count > 0 or right_count > 0:
        print(f"Lines detected - Left: {left_count}, Middle: {middle_count}, Right: {right_count}")
    if left_count == 0 and middle_count == 0 and right_count == 0:
        strike = 1
        print("All pins are down - Strike!")

    return left_count, middle_count, right_count

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    if bbox_fixed is None:
        results = model.predict(source=frame, show=False)

        if len(results[0].boxes) > 0:
            no_detection_count = 0  # Reset nodetection if it's see pinbowling
            last_frame = frame  # collect last frame 
            box = results[0].boxes[0].xyxy[0]
            x1, y1, x2, y2 = map(int, box)

            frame_count += 1
            if frame_count == fix_frame:
                bbox_fixed = (x1, y1, x2, y2)
                print(f"Bounding box fixed at frame {fix_frame}: {bbox_fixed}")
        else:
            no_detection_count += 1
            print(f"no detect : {no_detection_count}")
            if no_detection_count >= max_no_detection:
                if last_frame is not None:
                    frame = last_frame
                    print("Using last detected frame due to timeout.")

    if bbox_fixed is not None:
        x1, y1, x2, y2 = bbox_fixed
        roi = frame[y1:y2, x1:x2]

        # Count number left, middle, and right
        left_count, middle_count, right_count = count_standing_pins(roi)
        cv2.imshow("Detected Pins", roi)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
