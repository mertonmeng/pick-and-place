import copy
import cv2
import os
import numpy as np
import pykinect_azure as pykinect

from pupil_apriltags import Detector
from data_collection.calibration_3d_offline_data_collection import get_gripper_camera_3d_location
from utils.graphics_utils import draw_text

if __name__ == "__main__":

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
        break

    gripper_camera_3d_coords, point_3d_graphics_list, annotated_image = \
                get_gripper_camera_3d_location(color_image, transformed_depth_image, at_detector, device)
    
    R = np.load("calibration_3d_r.npy")
    t = np.load("calibration_3d_t.npy")
    gripper_camera_3d_coords_arr = np.array(gripper_camera_3d_coords).reshape(-1, 1)
    converted_robot_coords = np.matmul(R, gripper_camera_3d_coords_arr) + t[:,np.newaxis]
    draw_text(annotated_image, f"X: {converted_robot_coords[0, 0]:.2f}, Y: {converted_robot_coords[1, 0]:.2f}, Z: {converted_robot_coords[2, 0]:.2f}", 1000, 100)

    cv2.imshow("Annotated image", annotated_image)
    cv2.waitKey(0)