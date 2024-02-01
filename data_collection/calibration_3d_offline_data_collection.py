import copy
import cv2
import os
import numpy as np
import pykinect_azure as pykinect
import time
from pymycobot.mycobot import MyCobot

from utils.graphics_utils import draw_apriltag, get_3d_point_graphics, Open3dLiveVisualizer
from utils.geometry_utils import get_april_tags_2d_coords_gripper
from pupil_apriltags import Detector
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t

CALIB_PLATE_TO_GRIPPER_DEPTH_DIFF = 31
CALIBRATION_DATA_FOLDER_PATH = "calibration/3d_data/"

def generate_neighboring_points(center_x, center_y, size=5, step=1):
    """
    Generates a numpy array of neighboring points in a N x N grid around the given center point.

    :param center_x: x-coordinate of the center point
    :param center_y: y-coordinate of the center point
    :param size: size of the grid (default is N x N)
    :return: numpy array of neighboring points
    """
    offset = size // 2
    x_values = np.arange(center_x - offset, center_x + offset + 1, step)
    y_values = np.arange(center_y - offset, center_y + offset + 1, step)
    return np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

def get_gripper_camera_3d_location(color_image, transformed_depth_image, at_detector, device):
    annotated_image = copy.deepcopy(color_image)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    apriltags = at_detector.detect(grayscale_image)
    
    if len(apriltags) < 4:
        return []

    avg_depth = 0
    point_3d_graphics_list = []
    valid_depth_count = 0
    for tag in apriltags:
        draw_apriltag(annotated_image, tag)
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        sample_points = generate_neighboring_points(cX, cY, 10, 2)
        center_pixels = k4a_float2_t((cX, cY))
        center_rgb_depth = transformed_depth_image[cY, cX]

        center_pos3d_color = device.calibration.convert_2d_to_3d(
            center_pixels, 
            center_rgb_depth, 
            K4A_CALIBRATION_TYPE_COLOR, 
            K4A_CALIBRATION_TYPE_COLOR)

        center_pos3d_depth = device.calibration.convert_2d_to_3d(
            center_pixels, 
            center_rgb_depth, 
            K4A_CALIBRATION_TYPE_COLOR, 
            K4A_CALIBRATION_TYPE_DEPTH)

        point_3d_graphics = get_3d_point_graphics(
            center_pos3d_color.xyz.x,
            center_pos3d_color.xyz.y,
            center_pos3d_color.xyz.z
        )
        point_3d_graphics_list.append(point_3d_graphics)

        depth_list = []
        for idx in range(sample_points.shape[0]):
            pixel_x, pixel_y = sample_points[idx]
            point_depth = transformed_depth_image[pixel_y, pixel_x]
            depth_list.append(point_depth)
        depth_array = np.array(depth_list)
        non_zero_indices = np.nonzero(depth_array)
        filtered_depth_array = depth_array[non_zero_indices]
        median_depth = np.median(filtered_depth_array)
        print(f"Median Depth: {median_depth}")
        if median_depth != 0.0:
            avg_depth += median_depth
            valid_depth_count += 1
        
    avg_depth /= valid_depth_count
    estimated_gripper_depth = avg_depth - CALIB_PLATE_TO_GRIPPER_DEPTH_DIFF
    print(f"Avg Depth: {avg_depth}, Estimated Gripper Depth: {estimated_gripper_depth}")
    gripper_image_pixel_coords = get_april_tags_2d_coords_gripper(apriltags)
    
    gripper_X, gripper_Y, theta = tuple(map(int, gripper_image_pixel_coords))
    cv2.circle(annotated_image, (gripper_X, gripper_Y), 5, (0, 0, 255), -1)
    gripper_point_2d = k4a_float2_t((gripper_X, gripper_Y))

    gripper_pos3d_color = device.calibration.convert_2d_to_3d(
        gripper_point_2d, 
        center_rgb_depth, 
        K4A_CALIBRATION_TYPE_COLOR, 
        K4A_CALIBRATION_TYPE_COLOR)

    gripper_point_graphics = get_3d_point_graphics(
            gripper_pos3d_color.xyz.x,
            gripper_pos3d_color.xyz.y,
            estimated_gripper_depth
        )
    point_3d_graphics_list.append(gripper_point_graphics)

    gripper_camera_3d_coords = [
        gripper_pos3d_color.xyz.x,
        gripper_pos3d_color.xyz.y,
        estimated_gripper_depth
    ]
    return gripper_camera_3d_coords, point_3d_graphics_list, annotated_image

if __name__ == "__main__":

    mc = MyCobot("COM7", 115200)
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
    )

    x = 150.0
    y = -100.0
    z = 190
    open3dVisualizer = Open3dLiveVisualizer()

    while True:

        # Get capture
        capture = device.update()

        # Get the 3D point cloud
        ret_point, points = capture.get_transformed_pointcloud()

        # Get the color image in the depth camera axis
        ret_color, color_image = capture.get_color_image()

        # Get the transformed depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_point or not ret_depth:
            continue
        np.save(os.path.join(CALIBRATION_DATA_FOLDER_PATH, "color_image.npy"), color_image)
        np.save(os.path.join(CALIBRATION_DATA_FOLDER_PATH,"point_cloud.npy"), points)
        break

    idx = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                mc.send_coords([x + i * 20.0, y + j * 40.0, z + k*10.0, -180.0, 0.0, -90.0], 80, 1)
                if k == 0:
                    time.sleep(5)
                else:
                    time.sleep(2)

                # Get capture
                capture = device.update()

                # Get the 3D point cloud
                ret_point, points = capture.get_transformed_pointcloud()

                # Get the color image in the depth camera axis
                ret_color, color_image = capture.get_color_image()

                # Get the transformed depth
                ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

                if not ret_color or not ret_point or not ret_depth:
                    continue

                gripper_camera_3d_coords, point_3d_graphics_list, annotated_image = \
                    get_gripper_camera_3d_location(color_image, transformed_depth_image, at_detector, device)

                with open(os.path.join(CALIBRATION_DATA_FOLDER_PATH, "robot_coords_3d.txt"), "a") as f:
                    f.write(str(mc.get_coords()) + "\n")
                with open(os.path.join(CALIBRATION_DATA_FOLDER_PATH, "camera_coords_3d.txt"), "a") as f:
                    f.write(str(gripper_camera_3d_coords) + "\n")
                
                cv2.imshow("April tag image", annotated_image)
                # Initialize the Open3d visualizer

                open3dVisualizer(points, point_3d_graphics_list, color_image)
                time.sleep(1)
