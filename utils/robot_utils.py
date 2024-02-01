import time

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

def denormalize_robot_coord(coords_arr):
    coords_arr[:, 0] = coords_arr[:, 0] * ROBOT_X_COORD_NORMALIZER + ROBOT_X_COORD_ZERO_POINT
    coords_arr[:, 1] = coords_arr[:, 1] * ROBOT_Y_COORD_NORMALIZER + ROBOT_Y_COORD_ZERO_POINT
    coords_arr[:, 2] = coords_arr[:, 2] * ROBOT_THETA_NORMALIZER + ROBOT_THETA_ZERO_POINT
    return coords_arr

def normalize_camera_coord(cX, cY, theta, image_width, image_height):
    normX = (cX - image_width / 2) / (image_width / 2)
    normY = (cY - image_height / 2) / (image_height / 2)
    normTheta = (theta - IMAGE_THETA_ZERO_POINT) / IMAGE_THETA_NORMALIZER
    return normX, normY, normTheta

def pick_and_place(robot_controller, target_coords):
    # Move to target place
    robot_controller.send_coords([target_coords[0, 0], target_coords[0, 1], 220, -180.0, 0.0, target_coords[0, 2]], 30, 1)
    time.sleep(5)

    # Descend
    robot_controller.send_coords([target_coords[0, 0], target_coords[0, 1], 170, -180.0, 0.0, target_coords[0, 2]], 50, 1)
    time.sleep(2)

    # Grip
    robot_controller.set_gripper_mode(0)
    time.sleep(2)
    robot_controller.set_gripper_state(1, 50)
    time.sleep(2)

    # Pick up
    robot_controller.send_coords([target_coords[0, 0], target_coords[0, 1], 220, -180.0, 0.0, target_coords[0, 2]], 50, 1)
    time.sleep(2)

    # Move to target pose
    robot_controller.send_coords([200, 0, 190, -180.0, 0.0, -90.0], 50, 1)
    time.sleep(5)

    # Drop
    robot_controller.set_gripper_mode(0)
    time.sleep(2)
    robot_controller.set_gripper_state(0, 50)
    time.sleep(2)