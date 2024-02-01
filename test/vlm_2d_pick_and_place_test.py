import cv2
import numpy as np
import torch
import argparse
from utils.graphics_utils import draw_text
from calibration.models import Robot2DCalibrationNN, evaluate_model
import pykinect_azure as pykinect
from pymycobot.mycobot import MyCobot
from utils.vlm_detection_util import detect_object
from utils.robot_utils import *

IMAGE_THETA_ZERO_POINT = -130.0
IMAGE_THETA_NORMALIZER = 130.0

ROBOT_X_COORD_ZERO_POINT = 250.0
ROBOT_Y_COORD_NORMALIZER = 100.0
ROBOT_X_COORD_NORMALIZER = 50.0
ROBOT_Y_COORD_ZERO_POINT = 0.0
ROBOT_THETA_ZERO_POINT = -90.0
ROBOT_THETA_NORMALIZER = 90.0

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", help="Prompt for the object to pickup", required=True)
    parser.add_argument("--joint_angles_file", help="File containing joint angles", default="calibration/angles.txt")
    parser.add_argument("--coordinates_file", help="File containing 6DoF coordinates", default="calibration/coords.txt")
    parser.add_argument("--starting_id", help="Starting Tag Id", default=14)
    parser.add_argument("--origin_tag_id", help="Origin Tag Id", default=90)
    parser.add_argument("--calibration_model_path", help="Calibration Model path", default="calibration_model.pth")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mc = MyCobot("COM7", 115200)

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P

    # Start device
    device = pykinect.start_device(config=device_config)

    calibration_model = Robot2DCalibrationNN(3, 3)
    calibration_model.load_state_dict(torch.load(args.calibration_model_path))
    calibration_model.eval()
    calibration_model.to("cuda")

    ret = False
    while not ret:
    # Get capture
        capture = device.update()
        # Get the color image from the capture
        ret, color_image = capture.get_color_image()
    
    image_width = color_image.shape[1]
    image_height = color_image.shape[0]
    print(image_width, image_height)

    color_image, image_coords,_ = detect_object(args.query, color_image)
    if image_coords == None:
        print("Cannot get image coords")
        exit()
    cX, cY, theta = image_coords
    cv2.circle(color_image, (int(cX), int(cY)), 5, (0, 0, 255), -1)

    print(image_coords)

    normX, normY, normTheta = normalize_camera_coord(cX, cY, theta, image_width, image_height)

    input_array = np.array([[normX, normY, normTheta]]).astype(np.float32)

    predicted_val = evaluate_model(calibration_model, input_array)
    denorm_coords = denormalize_robot_coord(predicted_val)
    draw_text(color_image, f"X: {denorm_coords[0, 0]:.2f}, Y: {denorm_coords[0, 1]:.2f}, Theta: {denorm_coords[0, 2]:.2f}", 1000, 100)
    cv2.imshow('Calibration', color_image)
    cv2.waitKey(0)

    pick_and_place(mc, denorm_coords)