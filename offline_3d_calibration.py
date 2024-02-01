import ast
import os
import numpy as np
import torch
import open3d as o3d
import numpy as np
from scipy.linalg import svd

from utils.training_data_utils import split_list
from calibration.models import LinearRegression
from calibration.models import train_model, evaluate_model
from torch.optim.lr_scheduler import StepLR

CALIBRATION_DATA_FOLDER_PATH = "calibration/3d_data/"

def get_model_params():
    learningRate = 0.000005
    epochs = 2000

    model = LinearRegression(4, 4)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, 
                    step_size = 100, # Period of learning rate decay
                    gamma = 0.9) # Multiplicative factor of learning rate decay
    return model, criterion, optimizer, scheduler, epochs

def generate_calibration_coords_dataset_xyzw(
        camera_coords_list, 
        robot_coords_list):

    camera_coords_train_list, camera_coords_test_list, robot_coords_train_list, robot_coords_test_list \
        = split_list(camera_coords_list, robot_coords_list, 0.9)
        
    camera_coords_train = np.array(camera_coords_train_list).astype(np.float32)
    robot_coords_train = np.array(robot_coords_train_list).astype(np.float32)
    camera_coords_test = np.array(camera_coords_test_list).astype(np.float32)
    robot_coords_test = np.array(robot_coords_test_list).astype(np.float32)

    robot_coords_train = robot_coords_train[:, [0, 1, 2]]
    robot_coords_test = robot_coords_test[:, [0, 1, 2]]

    train_seq_len = robot_coords_train.shape[0]
    test_seq_len = robot_coords_test.shape[0]

    uni_vector_train = np.ones((train_seq_len, 1), dtype=np.float32)
    uni_vector_test = np.ones((test_seq_len, 1), dtype=np.float32)

    robot_coords_train = np.hstack((robot_coords_train, uni_vector_train))
    robot_coords_test = np.hstack((robot_coords_test, uni_vector_test))
    camera_coords_train = np.hstack((camera_coords_train, uni_vector_train))
    camera_coords_test = np.hstack((camera_coords_test, uni_vector_test))

    return camera_coords_train, robot_coords_train, camera_coords_test, robot_coords_test

def calibration_with_icp_registration(camera_coords_train, robot_coords_train):
        # Convert numpy arrays to Open3D PointClouds
    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(camera_coords_train)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_coords_train)

    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the bounding box to the visualizer
    vis.add_geometry(pcd_camera)
    pcd_camera.paint_uniform_color([1, 0, 0])
    vis.add_geometry(pcd_robot)
    pcd_robot.paint_uniform_color([0, 0, 1])

    # Run the visualizer
    vis.run()

    # ICP registration
    threshold = 10  # Set this to a suitable value
    trans_init = np.identity(4)  # Initial transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_camera, pcd_robot, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # Extract the transformation matrix
    transformation_matrix = reg_p2p.transformation
    print("Transformation Matrix:")
    print(transformation_matrix)
    return transformation_matrix

def calibration_with_machine_learning(camera_coords_train, robot_coords_train, camera_coords_test, robot_coords_test):
    model, criterion, optimizer, scheduler, epochs = get_model_params()

    model = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs,
        camera_coords_train, 
        robot_coords_train, 
        camera_coords_test, 
        robot_coords_test)
    
    return model

def centroid(X):
    return np.mean(X, axis=0)

def calibration_with_svd(A, B):
    A_centroid = centroid(A)
    B_centroid = centroid(B)
    A_centered = A - A_centroid
    B_centered = B - B_centroid
    U, _, Vt = svd(np.dot(B_centered.T, A_centered))
    R = np.dot(U, Vt)
    t = B_centroid - np.dot(R, A_centroid)
    return R, t

def generate_calibration_coords_dataset_xyz(
        camera_coords_list, 
        robot_coords_list):

    camera_coords_train_list, camera_coords_test_list, robot_coords_train_list, robot_coords_test_list \
        = split_list(camera_coords_list, robot_coords_list, 0.9)
        
    camera_coords_train = np.array(camera_coords_train_list).astype(np.float32)
    robot_coords_train = np.array(robot_coords_train_list).astype(np.float32)
    camera_coords_test = np.array(camera_coords_test_list).astype(np.float32)
    robot_coords_test = np.array(robot_coords_test_list).astype(np.float32)

    robot_coords_train = robot_coords_train[:, [0, 1, 2]]
    robot_coords_test = robot_coords_test[:, [0, 1, 2]]

    return camera_coords_train, robot_coords_train, camera_coords_test, robot_coords_test



if __name__ == "__main__":
    with open(os.path.join(CALIBRATION_DATA_FOLDER_PATH, "robot_coords_3d.txt")) as f:
        robot_coords_3d_txt = f.readlines()
    with open(os.path.join(CALIBRATION_DATA_FOLDER_PATH, "camera_coords_3d.txt")) as f:
        camera_coords_3d_txt = f.readlines()

    if len(robot_coords_3d_txt) != len(camera_coords_3d_txt):
        print("Robot coords and Camera coords length does not match")
        exit(1)

    i = 0
    camera_coords_3d_list = []
    robot_coords_3d_list = []
    while i < len(robot_coords_3d_txt):
        if "nan" in camera_coords_3d_txt[i]:
            camera_coords_3d_txt.pop(i)
            robot_coords_3d_txt.pop(i)
            continue

        camera_coords_3d = ast.literal_eval(camera_coords_3d_txt[i])
        robot_coords_3d = ast.literal_eval(robot_coords_3d_txt[i])
        i+=1
        if 0.0 in camera_coords_3d:
            continue
        camera_coords_3d_list.append(camera_coords_3d)
        robot_coords_3d_list.append(robot_coords_3d)

    # camera_coords_train, robot_coords_train, camera_coords_test, robot_coords_test \
    #     = generate_calibration_coords_dataset_xyzw(
    #         camera_coords_3d_list, 
    #         robot_coords_3d_list)
    
    camera_coords_train, robot_coords_train, camera_coords_test, robot_coords_test \
        = generate_calibration_coords_dataset_xyz(
            camera_coords_3d_list, 
            robot_coords_3d_list)

    # model = calibration_with_machine_learning(camera_coords_train,
    #                                           robot_coords_train,
    #                                           camera_coords_test,
    #                                           robot_coords_test)
    # predicted_val = evaluate_model(model, camera_coords_test)

    # Compute rotation and translation
    R, t = calibration_with_svd(camera_coords_train, robot_coords_train)

    # Print results
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(t)

    predicted_val = np.matmul(R, camera_coords_test.T) + t[:,np.newaxis]
    np.save("calibration_3d_r.npy", R)
    np.save("calibration_3d_t.npy", t)
    for i in range(robot_coords_test.shape[0]):
        print(f"Predicted Coords: {predicted_val[:, i].T}, Actual Coords: {robot_coords_test[i]}")

