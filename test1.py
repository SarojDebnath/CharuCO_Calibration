import cv2
import numpy as np
import json
import os

# Camera calibration matrix (replace with your actual calibration data)
fx, fy, cx, cy = 1000, 1000, 640, 360  # Replace with actual values
mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist = np.zeros((5,))  # Replace with actual distortion coefficients

# Charuco board parameters
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.01

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# JSON file to save the fixed position
FIXED_POSITION_FILE = "fixed_position.json"

# Variables to store fixed position
fixed_rvec = None
fixed_tvec = None

def save_fixed_position(frame):
    global fixed_rvec, fixed_tvec
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        if charuco_ids is not None and len(charuco_corners) > 3:
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, mtx, dist)
            if rvec is not None and tvec is not None:
                fixed_rvec = rvec
                fixed_tvec = tvec

                # Save to JSON
                data = {
                    "fixed_rvec": fixed_rvec.tolist(),
                    "fixed_tvec": fixed_tvec.tolist()
                }
                with open(FIXED_POSITION_FILE, 'w') as f:
                    json.dump(data, f, indent=4)
                print("Fixed position saved to", FIXED_POSITION_FILE)
                return True
    print("Failed to save fixed position.")
    return False

def calculate_relative_position(frame):
    global fixed_rvec, fixed_tvec
    if fixed_rvec is None or fixed_tvec is None:
        print("Fixed position not set.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        if charuco_ids is not None and len(charuco_corners) > 3:
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, mtx, dist)
            if rvec is not None and tvec is not None:
                # Calculate relative position
                relative_tvec = tvec - fixed_tvec
                relative_rvec = cv2.Rodrigues(rvec)[0] - cv2.Rodrigues(fixed_rvec)[0]

                # Convert translation to mm
                relative_tvec_mm = relative_tvec * 1000

                print("Relative Position (mm):", relative_tvec_mm.flatten())
                print("Relative Rotation (radians):", relative_rvec.flatten())

                # Draw pose
                cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)

def main():
    global fixed_rvec, fixed_tvec

    # Load fixed position if available
    if os.path.exists(FIXED_POSITION_FILE):
        with open(FIXED_POSITION_FILE, 'r') as f:
            data = json.load(f)
            fixed_rvec = np.array(data["fixed_rvec"], dtype=np.float32)
            fixed_tvec = np.array(data["fixed_tvec"], dtype=np.float32)
        print("Loaded fixed position from", FIXED_POSITION_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    print("Press 'S' to save fixed position, 'R' to calculate relative position, 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Show live feed
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save fixed position
            if save_fixed_position(frame):
                print("Fixed position saved.")
        elif key == ord('r'):  # Calculate relative position
            calculate_relative_position(frame)
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
