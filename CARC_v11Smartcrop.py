import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import serial
import numpy as np
points_list = []
green_select = []
points_select = []
pin_center = []

def load_image(cap):
    return cap.read()[1]

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    if brightness != 0:
        beta = brightness
        image = cv2.add(image, beta)
    return image

def detect_pins_with_yolo(model, image):
    # Perform detection using the YOLO model
    results = model(image)
    
    # Get the results from the model
    boxes = results[0].boxes.xywh  # YOLOv8 returns boxes in (x_center, y_center, width, height)
    
    return boxes

def refine_pin_detection_adjusted(model, img):
    x = y = x_center = y_center = new_x = new_y = new_h = new_w = rect_x = rect_y = rect_w = rect_h =  0
    smallest_white_field = None
    smallest_pincenter = None
    largest_green_point = 0
    distance = 0
    distance2 = 0

    # Detect pins using YOLOv8 model
    detected_boxes = detect_pins_with_yolo(model, img)

    # Initialize contour_frame to be the original image
    contour_frame = img.copy()
        
    # Use the first detected bounding box from YOLOv8 as the area for further processing
    if len(detected_boxes) > 0:
        # Take the first bounding box detected
        x_center, y_center, w, h = detected_boxes[0]
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        
        # Add 50 pixels to the width and height for cropping
        new_x = max(0, x - 180)
        new_y = max(0, y - 60)
        new_w = min(img.shape[1] - new_x, int(w + 220))  # W more tail increase
        new_h = min(img.shape[0] - new_y, int(h + 100))  # Ensure it doesn't go beyond the image boundaries

        # Replace manual slicing with YOLO-detected bounding box
        img = adjust_brightness_contrast(img, brightness=-15, contrast=21)
        sliced_image = img[new_y:new_y + new_h, new_x:new_x + new_w]
        
        # Draw the rectangle on the sliced image
        # Since the rectangle coordinates need to be relative to the cropped image, adjust accordingly
        rect_x = 180  # Offset added earlier to new_x for cropping
        rect_y = 60  # Offset added earlier to new_y for cropping
        rect_w = int(w)
        rect_h = int(h)

        cv2.rectangle(sliced_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
        cv2.putText(sliced_image, "Detected Pins", (rect_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        img = adjust_brightness_contrast(img, brightness=1, contrast=20)
        sliced_image = img[280:360, 280:520]
        
    # Proceed with the original HSV and Canny logic, but on `sliced_image`
    hsv = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([38, 10, 30])  
    upper_white = np.array([190, 108, 255])

    # # Define range of green color in HSV
    lower_green = np.array([60, 94, 70])
    upper_green = np.array([140, 205, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours for green pins
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_pin_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            green_pin_centers.append((center_x, center_y))
            cv2.circle(sliced_image, (center_x, center_y), 2, (0, 255, 0), -1)  # Green dot for green pin center

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        area = w * h

        epsilon = 0.08 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if area > 0:
            for point in approx:
                points_list.append((point[0][0], point[0][1]))
    
    # Rest of the original logic remains unchanged
    contour_frame = sliced_image.copy()

    gray = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    gray_inverted = cv2.bitwise_not(blur)
    _, binary = cv2.threshold(gray_inverted, 88, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    edges = cv2.Canny(binary, 60, 130)
    contours_pins, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 255, 0), 1)

    # Adjust the coordinates of the bounding box to cover the entire screen
    new_x = max(0, x-20)
    new_y = max(0, y-20)
    new_w = min(img.shape[1] - new_x, int(w + 180))  # Width with additional padding
    new_h = min(img.shape[0] - new_y, int(h + 120))  # Height with additional padding

    # Draw the red rectangle to cover the entire screen

    pin_contours = []
    for contour in contours_pins:
        area = cv2.contourArea(contour)
        if 0 < area < 1600:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if 0.4 < aspect_ratio < 5:
                pin_contours.append(contour)
                
                # Add a small red dot at the center of each pin contour
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(contour_frame, (center_x, center_y), 2, (0, 0, 255), -1)
    # Filter pins inside the adjusted rectangle and exclude areas near the edges
    edge_buffer = 7  # Buffer distance from the edge
    filtered_pins = [
        contour for contour in pin_contours 
        if new_x + edge_buffer  <= cv2.boundingRect(contour)[0] <= new_x + new_w - edge_buffer
        and new_y + edge_buffer <= cv2.boundingRect(contour)[1] <= new_y + new_h - edge_buffer 
    ]
    num_pins = len(filtered_pins)

    cv2.putText(contour_frame, f'Pins: {num_pins}', (14,  465), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
    for contour in filtered_pins:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(contour_frame, (center_x, center_y), 2, (0, 0, 255), -1)
        pin_center.append((center_x, center_y))
        smallest_pincenter = min(pin_center, key=lambda point: point[0]) 
           
    # SELECT YELLOW DOT EVERY CONTOUR/RED POINT SELECT IS SUITABLE  (White Pin)
    # for point in points_list:
    #     cv2.circle(contour_frame, (point[0], point[1]), 2, (0, 255, 255), -1)  # Yellow dots for contour points
    
    # for point in points_list:
    #     if new_x + 10 <= point[0] <= new_x + new_w and new_y <= point[1] <= new_y + new_h:
    #         points_select.append((point[0], point[1]))
    if points_list:
        # leftmost_point = min(points_list, key=lambda point: point[0])
        # bottommost_point = max([p for p in points_list if p[0] == leftmost_point[0]], key=lambda p: p[1])
        bottommost_point = max(points_list, key=lambda point: point[1])
        leftmost_point = min([p for p in points_list if p[1] == bottommost_point[1]], key=lambda p: p[0])
        cv2.circle(contour_frame, leftmost_point, 5, (0, 0, 255), -1) 
        
    # SELECT GREEN POINT IN BLUE RECTANGLE  (Green pin)
    # for point in green_pin_centers:
    #         green_select.append((point[0], point[1]))
    #         largest_green_select = max(green_select, key=lambda point: point[0])

                
    # Find the point with the smallest x value in (White Pin)
    if points_select:
        smallest_white_field = min(points_select, key=lambda point: point[0])
        cv2.circle(contour_frame, smallest_white_field, 5, (0, 0, 255), -1)

    # if len(green_pin_centers) != 0 and green_select[0] is not None and smallest_white_field is not None:
    #     if largest_green_select:
    #         pt1 = largest_green_select
    #     else:
    #         pt1 = green_select[0]            
    #     pt2 = smallest_white_field
    #     distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    #     cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)        
    #     cv2.putText(contour_frame, f'(S) Green: {distance:.2f}', (3, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
    # Case ( Pin White ) ##  -------------->  2
    if leftmost_point is not None:
        if len(detected_boxes) > 0:
            # Assuming rect_x and rect_y are the top-left corner coordinates of the rectangle
            pt1 = (rect_x, rect_y + rect_h)  # Convert rect_x and rect_y to a coordinate (x, y)
            pt2 = leftmost_point  # Ensure this is a coordinate tuple (x, y)

            # Calculate the Euclidean distance
            distance2 = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(contour_frame, f'(S) White: {distance2:.2f}', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
    # Case (1) || Case (2)
    comfirm = 0
    classify_found = '9'
    magnitude = 0
    # if len(green_pin_centers) != 0:
    #     if distance >= 100+magnitude and distance <= 120+magnitude and distance2 < 100:
    #         cv2.putText(contour_frame, f'A', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("A") 
    #         classify_found = "1"
    #         comfirm = 1           
    #     elif distance > 120 and distance <= 133:
    #         cv2.putText(contour_frame, f'B', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("B") 
    #         classify_found = "2"
    #         comfirm = 1
    #     elif distance > 133 and distance <= 150:
    #         cv2.putText(contour_frame, f'C', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("C") 
    #         classify_found = "3"
    #         comfirm = 1
    #     elif distance > 55 and distance <= 80 and distance2 >= 50 and distance2 < 70:
    #         cv2.putText(contour_frame, f'D', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("D") 
    #         classify_found = "5"
    #         comfirm = 1
    #     elif distance > 80 and distance <= 97 or distance2 >= 70 and distance2 < 88:
    #         cv2.putText(contour_frame, f'E', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("E") 
    #         classify_found = "6"
    #         comfirm = 1
    #     elif distance > 97 and distance <= 170 and distance2 > 88 and distance2 < 200:
    #         cv2.putText(contour_frame, f'F', (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #         print("F") 
    #         classify_found = "7"
    #         comfirm = 1
            
    # Assuming 'sliced_image' is the cropped frame
    screen_width = sliced_image.shape[1]  # Get the width of the sliced image

    # Define the boundaries for the left, middle, and right sections
    left_boundary = screen_width / 3
    right_boundary = 2 * screen_width / 3
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Calculate the total number of pixels in the left section
    left_section = white_mask[:, :int(left_boundary)]
    total_pixels = left_section.size
    white_pixels = cv2.countNonZero(left_section)

    # Calculate the percentage of white pixels in the left section
    white_percentage = (white_pixels / total_pixels) * 100
    
    # Determine the section of the bounding box based on rect_x (left edge of the bounding box)
    if rect_x < left_boundary:
        section = "left"
    elif rect_x < right_boundary:
        section = "middle"
    else:
        section = "right"

    mag = 7.5
    print(white_percentage)
    # Use the section information to adjust distance thresholds
    if section == "right":
        # Tuning for the right section
        print("Bounding box is in the right section.")
        if comfirm == 0 and distance2 is not None:
            if (white_percentage > 17) and distance2 > 170:
                cv2.putText(contour_frame, f'F', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("F")
                classify_found = "7"
            elif white_percentage < 6+mag or (distance2 >= 110 and distance2 <= 123):
                cv2.putText(contour_frame, f'A', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("A")
                classify_found = "1"
            elif (white_percentage >= 6+mag and white_percentage <= 9+mag) or (distance2 > 123 and distance2 <= 138):
                cv2.putText(contour_frame, f'B', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("B")
                classify_found = "2"
            elif (white_percentage > 9+mag and white_percentage < 17+mag) or (distance2 > 110 and distance2 <= 130):
                cv2.putText(contour_frame, f'C', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("C")
                classify_found = "3"
            elif (white_percentage >= 17+mag and white_percentage < 26+mag) or (distance2 > 130 and distance2 <= 150):
                cv2.putText(contour_frame, f'D', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("D")
                classify_found = "5"
            elif (white_percentage >= 26+mag and white_percentage <= 29.5+mag) or (distance2 > 50 and distance2 < 62):
                cv2.putText(contour_frame, f'E', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("E")
                classify_found = "6"
    else:
        # Tuning for the left and middle sections (default behavior)
        if comfirm == 0 and distance2 is not None:
            if white_percentage > 17 and distance2 > 182:
                cv2.putText(contour_frame, f'F', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("F")
                classify_found = "7"
            elif  white_percentage < 6+mag or (distance2 >= 110 and distance2 <= 122):
                cv2.putText(contour_frame, f'A', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("A")
                classify_found = "1"
            elif (white_percentage >= 6+mag and white_percentage <= 9+mag) or (distance2 > 122 and distance2 <= 134):
                cv2.putText(contour_frame, f'B', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("B")
                classify_found = "2"
            elif (white_percentage > 9+mag and white_percentage < 17+mag) or (distance2 > 134 and distance2 <= 153):
                cv2.putText(contour_frame, f'C', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("C")
                classify_found = "3"
            elif (white_percentage >= 17+mag and white_percentage < 24.5+mag) or (distance2 > 153 and distance2 <= 168):
                cv2.putText(contour_frame, f'D', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("D")
                classify_found = "5"
            elif (white_percentage >= 24.5+mag and white_percentage <= 31+mag) or (distance2 > 168 and distance2 < 182):
                cv2.putText(contour_frame, f'E', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                print("E")
                classify_found = "6"

    if classify_found == '9':
            classify_found = "3"

    return contour_frame, num_pins, sliced_image.shape , classify_found


# Load the YOLOv8 model (replace with your .pt file path)
yolo_model_path = 'models/pinbowling.pt'
model = YOLO(yolo_model_path)

cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("Failed to open /dev/video0, trying /dev/video1...")
    cap = cv2.VideoCapture("/dev/video1")

if not cap.isOpened():
    raise ValueError("Error: Could not open video capture.")

cv2.namedWindow('Refined Contours', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    processed_frame, num_pins, shape, classify_found  = refine_pin_detection_adjusted(model, frame)
    cv2.imshow('Refined Contours', processed_frame)
    print(f"Number of pins: {num_pins}")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
