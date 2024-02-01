import cv2
import numpy as np
import torch
import argparse
import os
import ast
import matplotlib.pyplot as plt
import time

from pupil_apriltags import Detector
from torch.optim.lr_scheduler import StepLR
from utils.graphics_utils import draw_apriltag
from utils.training_data_utils import split_list
from calibration.models import Robot2DCalibrationNN
from calibration.models import train_model, evaluate_model
from utils.geometry_utils import get_april_tags_2d_coords, get_april_tags_2d_coords_gripper
from pymycobot.mycobot import MyCobot
import pykinect_azure as pykinect

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

    parser.add_argument("--calibration_image_folder", help="Folder containing calibration images", default="calibration/images")
    parser.add_argument("--joint_angles_file", help="File containing joint angles", default="calibration/angles.txt")
    parser.add_argument("--coordinates_file", help="File containing 6DoF coordinates", default="calibration/coords.txt")
    parser.add_argument("--starting_id", help="Starting Tag Id", default=14)
    parser.add_argument("--origin_tag_id", help="Origin Tag Id", default=90)
    parser.add_argument("--calibration_model_path", help="Calibration Model path", default="calibration_model.pth")
    parser.add_argument("--offline_calibration", action='store_true', help="Whether to perform online calibration")

    return parser.parse_args()

def load_image_paths(folder_path):
    images_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images_list.append(os.path.join(folder_path, filename))
    return images_list

def load_value_from_txt_to_list(path):
    with open(path, 'r') as f:
        angles_str_lines = f.readlines()
    value_list = []
    for string in angles_str_lines:
        value_list.append(ast.literal_eval(string))
    return value_list

def generate_calibration_coords_dataset(
        input_coords_list, 
        gt_coords_list,
        img_width, 
        img_height):

    input_coords_train_list, input_coords_test_list, gt_coords_train_list, gt_coords_test_list \
        = split_list(input_coords_list, gt_coords_list, 0.9)
        
    input_coords_train = np.array(input_coords_train_list).astype(np.float32)
    gt_coords_train = np.array(gt_coords_train_list).astype(np.float32)
    input_coords_test = np.array(input_coords_test_list).astype(np.float32)
    gt_coords_test = np.array(gt_coords_test_list).astype(np.float32)

    gt_coords_train = gt_coords_train[:, [0, 1, 5]]
    gt_coords_test = gt_coords_test[:, [0, 1, 5]]

    input_coords_train[:, 0] = (input_coords_train[:, 0] - img_width / 2) / (img_width / 2)
    input_coords_train[:, 1] = (input_coords_train[:, 1] - img_height / 2) / (img_height / 2)
    input_coords_train[:, 2] = (input_coords_train[:, 2] - IMAGE_THETA_ZERO_POINT) / IMAGE_THETA_NORMALIZER

    input_coords_test[:, 0] = (input_coords_test[:, 0] - img_width / 2) / (img_width / 2)
    input_coords_test[:, 1] = (input_coords_test[:, 1] - img_height / 2) / (img_height / 2)
    input_coords_test[:, 2] = (input_coords_test[:, 2] - IMAGE_THETA_ZERO_POINT) / IMAGE_THETA_NORMALIZER

    gt_coords_train[:, 0] = (gt_coords_train[:, 0] - ROBOT_X_COORD_ZERO_POINT) / ROBOT_X_COORD_NORMALIZER
    gt_coords_train[:, 1] = (gt_coords_train[:, 1] - ROBOT_Y_COORD_ZERO_POINT) / ROBOT_Y_COORD_NORMALIZER
    gt_coords_train[:, 2] = (gt_coords_train[:, 2] - ROBOT_THETA_ZERO_POINT) / ROBOT_THETA_NORMALIZER

    gt_coords_test[:, 0] = (gt_coords_test[:, 0] - ROBOT_X_COORD_ZERO_POINT) / ROBOT_X_COORD_NORMALIZER
    gt_coords_test[:, 1] = (gt_coords_test[:, 1] - ROBOT_Y_COORD_ZERO_POINT) / ROBOT_Y_COORD_NORMALIZER
    gt_coords_test[:, 2] = (gt_coords_test[:, 2] - ROBOT_THETA_ZERO_POINT) / ROBOT_THETA_NORMALIZER

    return input_coords_train, gt_coords_train, input_coords_test, gt_coords_test

def get_model_params():
    learningRate = 0.01
    epochs = 20000

    model = Robot2DCalibrationNN(3, 3)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, 
                    step_size = 100, # Period of learning rate decay
                    gamma = 0.99) # Multiplicative factor of learning rate decay
    return model, criterion, optimizer, scheduler, epochs

def denormalize(coords_arr):
    coords_arr[:, 0] = coords_arr[:, 0] * ROBOT_X_COORD_NORMALIZER + ROBOT_X_COORD_ZERO_POINT
    coords_arr[:, 1] = coords_arr[:, 1] * ROBOT_Y_COORD_NORMALIZER + ROBOT_Y_COORD_ZERO_POINT
    coords_arr[:, 2] = coords_arr[:, 2] * ROBOT_THETA_NORMALIZER + ROBOT_THETA_ZERO_POINT
    return coords_arr

def draw_histogram():
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    # Create histogram for each dataset on corresponding subplot
    axs[0, 0].hist(image_coords_train[:, 0], bins=50)
    axs[0, 1].hist(image_coords_train[:, 1], bins=50)
    axs[0, 2].hist(image_coords_train[:, 2], bins=50)
    axs[1, 0].hist(robot_coords_train[:, 0], bins=50)
    axs[1, 1].hist(robot_coords_train[:, 1], bins=50)
    axs[1, 2].hist(robot_coords_train[:, 2], bins=50)

    # Add titles and labels
    axs[0, 0].set_title('Image X Coords')
    axs[0, 1].set_title('Image Y Coords')
    axs[0, 2].set_title('Image Theta')
    axs[1, 0].set_title('Robot X Coords')
    axs[1, 1].set_title('Robot Y Coords')
    axs[1, 2].set_title('Robot Theta')

    # Add common x and y labels
    fig.text(0.5, 0.04, 'Value', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Show plot
    plt.show()

def collect_online_calibration_data():
    mc = MyCobot("COM7", 115200)

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

    # Start device
    device = pykinect.start_device(config=device_config)

    ret = False
    while not ret:
        # Get capture
        capture = device.update()
        # Get the color image from the capture
        ret, color_image = capture.get_color_image()

    theta_robot = -180.0
    x = 200.0
    y = -100.0

    robot_coords_list = []
    image_coords_list = []
    joint_angles_list = []

    cv2.namedWindow("Calib Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Raw Image", cv2.WINDOW_NORMAL)
    for i in range(6):
        for j in range(6):
            for k in range(13):
                mc.send_coords([x + i * 20.0, y + j * 40.0, 190, -180.0, 0.0, theta_robot + k * 15.0], 80, 1)
                if k == 0:
                    time.sleep(5)
                else:
                    time.sleep(2)
                # Get capture
                capture = device.update()
                # Get the color image from the capture
                ret, color_image = capture.get_color_image()

                if not ret:
                    continue
                cv2.imshow("Raw Image", color_image)
                grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                apriltags = at_detector.detect(grayscale_image)

                for tag in apriltags:
                    draw_apriltag(color_image, tag)

                if len(apriltags) < 4:
                    continue
                
                image_coords = get_april_tags_2d_coords_gripper(apriltags)
                if image_coords == None:
                    continue
                cX, cY, theta = image_coords
                cv2.circle(color_image, (int(cX), int(cY)), 5, (0, 0, 255), -1)
                
                cv2.imshow("Calib Image", color_image)
                cv2.waitKey(1000)
                robot_coords = mc.get_coords()
                print(f"Robot Coords: {robot_coords}")
                print(f"Image Coords: {image_coords}")
                time.sleep(1)
                joint_angles = mc.get_angles()
                time.sleep(1)
                if len(robot_coords) == 0 or len(joint_angles) == 0:
                    continue
                robot_coords_list.append(robot_coords)
                joint_angles_list.append(joint_angles)
                image_coords_list.append([cX, cY, theta])

    return image_coords_list, robot_coords_list, joint_angles_list

if __name__ == "__main__":
    args = parse_args()

    at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)

    if args.offline_calibration:
        images_list = load_image_paths(args.calibration_image_folder)

        image_coords_list = [None] * len(images_list)

        for image_filename in images_list:
            color_image = cv2.imread(image_filename)
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            apriltags = at_detector.detect(grayscale_image)
            for apriltag in apriltags:
                draw_apriltag(color_image, apriltag)
            cX, cY, theta = get_april_tags_2d_coords(apriltags)
            image_coords_list[int(os.path.basename(image_filename).split('.')[0])] = [cX, cY, theta]

        joint_angles_list = load_value_from_txt_to_list(args.joint_angles_file)
        robot_coords_list = load_value_from_txt_to_list(args.coordinates_file)
    else:
        image_coords_list, robot_coords_list, joint_angles_list = collect_online_calibration_data()
    
    image_coords_train, robot_coords_train, image_coords_test, robot_coords_test \
        = generate_calibration_coords_dataset(
            image_coords_list, 
            robot_coords_list, 
            IMAGE_WIDTH,
            IMAGE_HEIGHT)

    draw_histogram()

    model, criterion, optimizer, scheduler, epochs = get_model_params()

    model = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs,
        image_coords_train, 
        robot_coords_train, 
        image_coords_test, 
        robot_coords_test)
    predicted_val = evaluate_model(model, image_coords_test)
    denorm_predicted_val = denormalize(predicted_val)
    denorm_robot_coords_test = denormalize(robot_coords_test)
    for i in range(predicted_val.shape[0]):
        print(f"Predicted Coords: {denorm_predicted_val[i]}, Actual Coords: {denorm_robot_coords_test[i]}")
    
    torch.save(model.state_dict(), args.calibration_model_path)
