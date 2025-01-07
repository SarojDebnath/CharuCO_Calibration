import cv2
import numpy as np
import json

# Load calibration data
json_file_path = './calibration.json'
with open(json_file_path, 'r') as file: 
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dist = np.array(json_data['dist'])

# Charuco board parameters
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.01

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Global variables to store the fixed position
# Load fixed position from JSON if available
try:
    with open("fixed_position.json", "r") as json_file:
        fixed_data = json.load(json_file)
        fixed_rvec = np.array(fixed_data["fixed_rvec"])
        fixed_tvec = np.array(fixed_data["fixed_tvec"])
    print("Fixed position loaded from file.")
except FileNotFoundError:
    fixed_rvec = None
    fixed_tvec = None
    print("No fixed position found. Please save a fixed position.")


def save_fixed_position(frame):
    global fixed_rvec, fixed_tvec

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if len(marker_ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        if charuco_ids is not None and len(charuco_corners) > 3:
            # Initialize rvec and tvec
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, mtx, dist, rvec, tvec
            )

            if retval:
                fixed_rvec = rvec
                fixed_tvec = tvec
                # Save fixed position to JSON
                fixed_position = {
                    "fixed_rvec": fixed_rvec.tolist(),
                    "fixed_tvec": fixed_tvec.tolist()
                }
                with open("fixed_position.json", "w") as json_file:
                    json.dump(fixed_position, json_file, indent=4)
                print("Fixed position saved!")
                return True
            else:
                print("Failed to estimate pose.")
    print("Charuco board not detected.")
    return False


def calculate_relative_position(frame):
    global fixed_rvec, fixed_tvec

    if fixed_rvec is None or fixed_tvec is None:
        print("Fixed position not saved. Please save it first.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if len(marker_ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        if charuco_ids is not None and len(charuco_corners) > 3:
            # Initialize rvec and tvec
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, mtx, dist, rvec, tvec
            )

            if retval:
                # Calculate relative position
                relative_tvec = tvec - fixed_tvec
                relative_rmat, _ = cv2.Rodrigues(rvec - fixed_rvec)
                relative_rvec, _ = cv2.Rodrigues(relative_rmat)

                print(f"Relative Translation (X, Y, Z): {relative_tvec.flatten()}")
                print(f"Relative Rotation Vector: {relative_rvec.flatten()}")
                return relative_tvec, relative_rvec
            else:
                print("Failed to estimate pose.")
    print("Charuco board not detected.")
    return None

# Main loop for capturing and processing frames
cap = cv2.VideoCapture(0)  # Change 0 to your camera ID if needed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save the fixed position
        if save_fixed_position(frame):
            print("Fixed position saved successfully.")
        else:
            print("Failed to detect Charuco board. Try again.")

    elif key == ord('r'):  # Press 'r' to calculate the relative position
        calculate_relative_position(frame)

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
