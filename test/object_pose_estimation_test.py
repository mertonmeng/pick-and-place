import cv2
import open3d as o3d
import numpy as np
import pykinect_azure as pykinect
from utils.vlm_detection_util import detect_object
from scipy.optimize import least_squares

QUERY = "Find the charger"

import numpy as np

def error_function_least_squares(p, points):
    """ 
    Calculate the residuals for least squares fitting. 
    Each residual is the perpendicular distance from a point to the plane.
    """
    plane_normal = np.array(p[:3])
    plane_point = np.array(p[3:])
    residuals = []
    for point in points:
        # Calculate the vector from a point on the plane to the current point
        point_vector = point - plane_point
        # Calculate the distance from the point to the plane
        distance = np.dot(plane_normal, point_vector) / np.linalg.norm(plane_normal)
        residuals.append(distance)
    return residuals

def fit_plane_to_points(points):
    """ Fit a plane to the given 3D points using least squares. """
    # Initial guess for the plane: normal vector (1, 1, 1) and a point (0, 0, 0)
    initial_guess = [1, 1, 1, 0, 0, 0]
    # Use least squares to minimize the residuals
    result = least_squares(error_function_least_squares, initial_guess, args=[points])

    plane_normal = result.x[:3]
    plane_point = result.x[3:]
    # Extracting the normal vector components and the point on the plane
    n_x, n_y, n_z = plane_normal
    x0, y0, z0 = plane_point

    # Coefficients A, B, C are the components of the normal vector
    A, B, C = n_x, n_y, n_z

    # Coefficient D can be calculated using the point on the plane
    D = -(A * x0 + B * y0 + C * z0)

    return (A, B, C, D)

class Open3dBlockingVisualizer():

    def __init__(self):

        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def __call__(self, points_3d, oriented_bounding_box, rgb_image=None):

        self.draw(points_3d, oriented_bounding_box, rgb_image)

    def draw(self, points_3d, oriented_bounding_box, rgb_image=None):

        # Add values to vectors
        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        if rgb_image is not None:
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # oriented_bounding_box.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.vis.add_geometry(oriented_bounding_box)
            self.o3d_started = True
            self.vis.run()

def transform_point_to_3d(device, transformed_depth_image, cX, cY):
    rgb_depth = transformed_depth_image[cX, cY]
    pixels = pykinect.k4a_float2_t((cX, cY))

    pos3d_color_raw = device.calibration.convert_2d_to_3d(
        pixels, 
        rgb_depth, 
        pykinect.K4A_CALIBRATION_TYPE_COLOR, 
        pykinect.K4A_CALIBRATION_TYPE_COLOR)
    
    pos3d_color = (pos3d_color_raw.xyz.x, pos3d_color_raw.xyz.y, pos3d_color_raw.xyz.z)

    pos3d_depth_raw = device.calibration.convert_2d_to_3d(
        pixels, 
        rgb_depth, 
        pykinect.K4A_CALIBRATION_TYPE_COLOR, 
        pykinect.K4A_CALIBRATION_TYPE_DEPTH)
    
    pos3d_depth = (pos3d_depth_raw.xyz.x, pos3d_depth_raw.xyz.y, pos3d_depth_raw.xyz.z)

    return pos3d_color, pos3d_depth

def generate_neighboring_points(center_x, center_y, size=5):
    """
    Generates a numpy array of neighboring points in a 5x5 grid around the given center point.

    :param center_x: x-coordinate of the center point
    :param center_y: y-coordinate of the center point
    :param size: size of the grid (default is 5x5)
    :return: numpy array of neighboring points
    """
    offset = size // 2
    x_values = np.arange(center_x - offset, center_x + offset + 1)
    y_values = np.arange(center_y - offset, center_y + offset + 1)
    return np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

def create_plane_mesh(center, plane_params, bounds, resolution=50):
    """ Create a mesh for the plane. """
    # Unpack the plane parameters and bounds
    a, b, c, d = plane_params
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Create a grid of x, y points
    x = np.linspace(center[0] + x_min, center[0] + x_max, resolution)
    y = np.linspace(center[1] + y_min, center[1] + y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate corresponding z for the plane
    Z = (-d - a * X - b * Y) / c

    # Create vertices and triangles for the mesh
    vertices = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            triangles.append([idx + resolution, idx + 1, idx])
            triangles.append([idx + resolution, idx + 1 + resolution, idx + 1])
    triangles = np.array(triangles)

    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    return mesh

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
    i = 0
    while True:

        # Get capture
        capture = device.update()

        # Get the 3D point cloud
        ret_point, points = capture.get_transformed_pointcloud()

        # Get the color image in the depth camera axis
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_point or not ret_depth:
            continue
        
        i+=1
        if i == 100:
            break
    
    rgb_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    annotated_image, pose, box, mask_image = detect_object(QUERY, rgb_color_image)

    np.save("annotated_image.npy", annotated_image)
    np.save("mask_image.npy", mask_image)
    np.save("point_cloud.npy", points)

    if False:
        flatten_mask = mask_image.flatten() !=0
        selected_points = points[flatten_mask]
    else:
        # Create a blank mask with the same size as the image
        mask_image = np.zeros_like(rgb_color_image[:, :, 0])

        # Draw the rotated rectangle on the mask
        cv2.fillPoly(mask_image, [box], 1)

        flatten_mask = mask_image.flatten() !=0
        selected_points = points[flatten_mask]

    # Create a boolean mask where True indicates a non-zero row
    mask = ~(np.all(selected_points == 0, axis=1))

    # Apply the mask to filter out zero rows
    selected_points = selected_points[mask]

    # Example points
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(selected_points)
    obj_pcd.paint_uniform_color([1, 0, 0])  # Red color for points

    obb = obj_pcd.get_minimal_oriented_bounding_box()
    obb.color = np.array([0,1,0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = cv2.cvtColor(annotated_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Initialize the Open3d visualizer
    open3dVisualizer = Open3dBlockingVisualizer()

    open3dVisualizer(points, obb, annotated_image)