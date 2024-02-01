import cv2
import numpy as np
import torch
import argparse
import time
from pupil_apriltags import Detector
from utils.geometry_utils import get_april_tags_2d_coords_target
from utils.graphics_utils import draw_apriltag, draw_text
from calibration.models import Robot2DCalibrationNN
from calibration.models import evaluate_model
import pykinect_azure as pykinect
from pymycobot.mycobot import MyCobot

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

    parser.add_argument("--joint_angles_file", help="File containing joint angles", default="calibration/angles.txt")
    parser.add_argument("--coordinates_file", help="File containing 6DoF coordinates", default="calibration/coords.txt")
    parser.add_argument("--starting_id", help="Starting Tag Id", default=14)
    parser.add_argument("--origin_tag_id", help="Origin Tag Id", default=90)
    parser.add_argument("--calibration_model_path", help="Calibration Model path", default="calibration_model.pth")

    return parser.parse_args()

def denormalize(coords_arr):
    coords_arr[:, 0] = coords_arr[:, 0] * ROBOT_X_COORD_NORMALIZER + ROBOT_X_COORD_ZERO_POINT
    coords_arr[:, 1] = coords_arr[:, 1] * ROBOT_Y_COORD_NORMALIZER + ROBOT_Y_COORD_ZERO_POINT
    coords_arr[:, 2] = coords_arr[:, 2] * ROBOT_THETA_NORMALIZER + ROBOT_THETA_ZERO_POINT
    return coords_arr

if __name__ == "__main__":
    args = parse_args()

    mc = MyCobot("COM7", 115200)

    at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)

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

    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    apriltags = at_detector.detect(grayscale_image)
    for apriltag in apriltags:
        draw_apriltag(color_image, apriltag)
    if len(apriltags) < 2:
        print("Not Enough Tags")
        exit()
    image_coords = get_april_tags_2d_coords_target(apriltags)
    if image_coords == None:
        print("Cannot get image coords")
        exit()
    cX, cY, theta = image_coords
    cv2.circle(color_image, (int(cX), int(cY)), 5, (0, 0, 255), -1)

    print(image_coords)

    normX = (cX - image_width / 2) / (image_width / 2)
    normY = (cY - image_height / 2) / (image_height / 2)
    normTheta = (theta - IMAGE_THETA_ZERO_POINT) / IMAGE_THETA_NORMALIZER

    input_array = np.array([[normX, normY, normTheta]]).astype(np.float32)

    predicted_val = evaluate_model(calibration_model, input_array)
    denorm_coords = denormalize(predicted_val)
    draw_text(color_image, f"X: {denorm_coords[0, 0]:.2f}, Y: {denorm_coords[0, 1]:.2f}, Theta: {denorm_coords[0, 2]:.2f}", 1000, 100)
    # cv2.imshow('Calibration', color_image)

    # Move to target place
    mc.send_coords([denorm_coords[0, 0], denorm_coords[0, 1], 220, -180.0, 0.0, denorm_coords[0, 2]], 30, 1)
    time.sleep(5)

    # Descend
    mc.send_coords([denorm_coords[0, 0], denorm_coords[0, 1], 170, -180.0, 0.0, denorm_coords[0, 2]], 50, 1)
    time.sleep(2)

    # Grip
    mc.set_gripper_mode(0)
    time.sleep(2)
    mc.set_gripper_state(1, 50)
    time.sleep(2)

    # Pick up
    mc.send_coords([denorm_coords[0, 0], denorm_coords[0, 1], 220, -180.0, 0.0, denorm_coords[0, 2]], 50, 1)
    time.sleep(2)

    mc.send_coords([200, 0, 190, -180.0, 0.0, -90.0], 50, 1)
    time.sleep(5)

    mc.set_gripper_mode(0)
    time.sleep(2)
    mc.set_gripper_state(0, 50)
    time.sleep(2)

    # Press q key to stop
    # cv2.waitKey(0)
