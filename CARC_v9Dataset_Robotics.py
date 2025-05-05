import time
import numpy as np
import cv2
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

def refine_pin_detection_adjusted(img):
    smallest_pincenter = None
    smallest_white_field = None
    largest_green_point = 0
    distance = 0
    distance2 = 0
    img = adjust_brightness_contrast(img, brightness=2, contrast=21)
    sliced_image = img[290:380, 300:500]
    # frame_width, frame_height = img.shape[1], img.shape[0]
    # zoom_factor = 2
    # zoomed_frame = zoom(img, zoom_factor)

    # zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    # crop_x = (zoomed_width - frame_width) // 2 + 120
    # crop_y = (zoomed_height - frame_height) // 2 + 50

    # # Ensure cropping coordinates are within valid bounds
    # crop_x = max(crop_x, 0)
    # crop_y = max(crop_y, 0)
    # end_x = min(crop_x + frame_width, zoomed_width)
    # end_y = min(crop_y + frame_height, zoomed_height)

    # cropped_frame = zoomed_frame[crop_y:end_y, crop_x:end_x]
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 70])  
    upper_white = np.array([60, 25, 94])
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([70, 134, 70])
    upper_green = np.array([140, 205, 255])
    
    # # Define range of green color in HSV
    # lower_green = np.array([85, 90, 70])
    # upper_green = np.array([115, 205, 255])
    
    # Threshold the HSV image to get only green colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours for green pins
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        area = w * h

        epsilon = 0.08 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if area > 0:
            for point in approx:
                points_list.append((point[0][0], point[0][1]))
                # cv2.circle(sliced_image, (point[0][0], point[0][1]), 5, (0, 0, 255), -1)
    # Initialize contour_frame
    contour_frame = sliced_image.copy()
            
    gray = cv2.cvtColor(sliced_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    gray_inverted = cv2.bitwise_not(blur)
    _, binary = cv2.threshold(gray_inverted, 60, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray_inverted, 115, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    edges = cv2.Canny(binary, 30, 210)
    contours_pins, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 255, 0), 1)

    # Adjust the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contours_field[0])
    new_x = x
    new_y = y 
    new_w = w 
    new_h = h
    
    new_x2 = x
    new_y2 = y
    new_w2 = w 
    new_h2 = h
    
    cv2.rectangle(contour_frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)
    cv2.rectangle(contour_frame, (new_x2, new_y2), (new_x2 + new_w2, new_y2 + new_h2), (0, 0, 255), 2)
        
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
           
    # SELECT RED POINT IN BLUE RECTANGLE  (White Pin)
    for point in points_list:
        if new_x + 10 <= point[0] <= new_x + new_w and new_y <= point[1] <= new_y + new_h:
            points_select.append((point[0], point[1]))

    # SELECT GREEN POINT IN BLUE RECTANGLE  (Green pin)
    for point in green_pin_centers:
        # if new_x2 + 50 <= point[0] <= new_x2 + new_w2 and new_y2 <= point[1] <= new_y + new_h:
            green_select.append((point[0], point[1]))
                
    # Find the point with the smallest x value in (White Pin)

    if points_select:
        smallest_white_field = min(points_select, key=lambda point: point[0])
        cv2.circle(contour_frame, smallest_white_field, 5, (0, 0, 255), -1)
    # print(green_select)
    # Find the point with the largest x value in (Green Pin)
    if green_select:
        largest_green_point = max(green_select, key=lambda point: point[0])
        cv2.circle(sliced_image, largest_green_point, 5, (0, 255, 0), -1)  # Green dot for green pin center
    
    # print(f"small = {smallest_white_field}")
    # print(f"large = {largest_green_point}")
    # Case ( Pin Green ) ##  -------------->  1
    # print(len(green_pin_centers))
    # print(smallest_pincenter)
    if len(green_pin_centers) != 0 and green_select[0] is not None and smallest_white_field is not None:
        pt1 = largest_green_point
        pt2 = smallest_white_field
        distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)        
        cv2.putText(contour_frame, f'(S) Green: {distance:.2f}', (3, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
    # Case ( Pin White ) ##  -------------->  2
    if len(pin_center) != 0 and smallest_pincenter is not None and smallest_white_field is not None:
        if len(points_list) >= 2:
            pt1 = smallest_pincenter
            pt2 = smallest_white_field
            distance2 = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(contour_frame, f'(S) White: {distance2:.2f}', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
    # Case (1) || Case (2)
    comfirm = 0
    classify_found = '9'
    if len(green_pin_centers) != 0:
        if distance >= 95 and distance <= 112.5 and distance2 < 100:
            cv2.putText(contour_frame, f'A', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("A") 
            classify_found = "1"
            comfirm = 1         
        elif distance > 112.5 and distance <= 131:
            cv2.putText(contour_frame, f'B', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("B") 
            classify_found = "2"
            comfirm = 1
        elif distance > 131 and distance <= 156:
            cv2.putText(contour_frame, f'C', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("C") 
            classify_found = "3"
            comfirm = 1
        elif distance > 70 and distance <= 85:
            cv2.putText(contour_frame, f'D', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("D") 
            classify_found = "5"
            comfirm = 1
        elif (distance > 85 and distance <= 96) or distance2 > 129 and distance < 145:
            cv2.putText(contour_frame, f'E', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("E") 
            classify_found = "6"
            comfirm = 1
        elif (distance > 96 and distance <= 112) or distance >= 160 or distance2 > 145 and distance2 < 200:
            cv2.putText(contour_frame, f'F', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("F") 
            classify_found = "7"
            comfirm = 1
            
    if comfirm == 0 and len(pin_center) != 0:
        if distance2 >= 50 and distance2 <= 82:
            cv2.putText(contour_frame, f'A', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("A") 
            classify_found = "1"
        elif distance2 > 82 and distance2 <= 94:
            cv2.putText(contour_frame, f'B', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("B") 
            classify_found = "2"
        elif distance2 > 94 and distance2 <= 107:
            cv2.putText(contour_frame, f'C', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("C")
            classify_found = "3"
        elif distance2 > 107 and distance2 <= 130:
            cv2.putText(contour_frame, f'D', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("D")
            classify_found = "5"
        elif distance2 > 130 and distance2 <= 160:
            cv2.putText(contour_frame, f'E', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("E")
            classify_found = "6"
        elif distance2 > 160:
            cv2.putText(contour_frame, f'F', (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print("F")
            classify_found = "7"

    if classify_found == '9':
            classify_found = "3"    	    
    # # Draw bounding boxes for all detected pin contours
    # for contour in pin_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green for all detected pins

    # # Draw bounding boxes for filtered pins
    # for contour in filtered_pins:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for filtered pins

    return contour_frame, num_pins, sliced_image.shape , classify_found


def main():
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow time for the camera to warm up

    while True:
        image = load_image(cap)
        if image is not None:
            adjusted_contour_frame, adjusted_num_pins, shape, classify_found = refine_pin_detection_adjusted(image)
            cv2.imshow('Refined Pin Detection', adjusted_contour_frame)  # Show the image

            print(classify_found)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
