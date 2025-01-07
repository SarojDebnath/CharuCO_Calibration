import cv2
import numpy as np
import json
import time

# Load the camera calibration data
json_file_path = './calibration.json'

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])  # Camera matrix
dist = np.array(json_data['dist'])  # Distortion coefficients

# Fixed position variables
fixed_rvec = None
fixed_tvec = None

# Load the Charuco board
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.01

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Convert displacements to mm
def convert_to_mm(value_meters):
    return value_meters * 1000

# Save the fixed position to a JSON file
def save_fixed_position_to_json(rvec, tvec):
    fixed_position_data = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist()
    }
    with open('fixed_position.json', 'w') as file:
        json.dump(fixed_position_data, file, indent=4)

# Calculate relative position and display it
def calculate_relative_position(frame):
    global fixed_rvec, fixed_tvec

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if len(marker_ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

        if charuco_ids is not None and len(charuco_corners) > 3:
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, mtx, dist)

            if fixed_rvec is not None and fixed_tvec is not None:
                # Compute displacement
                displacement = tvec - fixed_tvec
                displacement_mm = convert_to_mm(displacement)

                # Draw axes and display displacement
                cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)
                text = f"X: {displacement_mm[0][0]:.2f} mm, Y: {displacement_mm[1][0]:.2f} mm, Z: {displacement_mm[2][0]:.2f} mm"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save the fixed position
def save_fixed_position(frame):
    global fixed_rvec, fixed_tvec

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if len(marker_ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

        if charuco_ids is not None and len(charuco_corners) > 3:
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, mtx, dist)
            if rvec is not None and tvec is not None:
                fixed_rvec = rvec
                fixed_tvec = tvec
                save_fixed_position_to_json(rvec, tvec)
                return True
    return False

# Main loop
cap = cv2.VideoCapture(0)  # Change to your camera ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # Save fixed position
    if key == ord('s'):
        if save_fixed_position(frame):
            print("Fixed position saved successfully.")
        else:
            print("Failed to save fixed position.")

    # Calculate relative position
    elif key == ord('r'):
        calculate_relative_position(frame)

    # Exit
    elif key == ord('q'):
        break

    cv2.imshow("Charuco Tracker", frame)

cap.release()
cv2.destroyAllWindows()
