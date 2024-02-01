import cv2
import open3d as o3d
import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import least_squares

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

annotated_image = np.load("annotated_image.npy")
cv2.imshow("color image", annotated_image)
cv2.waitKey(0)
mask_image = np.load("mask_image.npy")
cv2.imshow("mask image", mask_image)
cv2.waitKey(0)
points = np.load("point_cloud.npy")

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

obb = obj_pcd.get_oriented_bounding_box()
obb.color = np.array([0,1,0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = cv2.cvtColor(annotated_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
pcd.colors = o3d.utility.Vector3dVector(colors)

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the bounding box to the visualizer
vis.add_geometry(obb)
vis.add_geometry(pcd)

# Run the visualizer
vis.run()