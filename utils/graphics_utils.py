import cv2
import open3d as o3d

class Open3dLiveVisualizer():

    def __init__(self):

        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def __call__(self, points_3d, other_graphics, rgb_image=None):

        self.update(points_3d, other_graphics, rgb_image)

    def update(self, points_3d, other_graphics, rgb_image=None):

        # Add values to vectors
        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        if rgb_image is not None:
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            for graphics in other_graphics:
                self.vis.add_geometry(graphics)
            self.o3d_started = True
        else:
            for graphics in other_graphics:
                self.vis.update_geometry(graphics)
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

class Open3dBlockingVisualizer():
    def __init__(self):

        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def __call__(self, points_3d, other_graphics, rgb_image=None):

        self.draw(points_3d, other_graphics, rgb_image)

    def draw(self, points_3d, other_graphics, rgb_image=None):

        # Add values to vectors
        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        if rgb_image is not None:
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Add geometries if it is the first time
        for graphics in other_graphics:
            self.vis.add_geometry(graphics)
        self.vis.add_geometry(self.point_cloud)
        self.vis.run()

def get_3d_point_graphics(x, y, z):
    point = [x, y, z]
    # Create a small sphere at the point location
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    sphere.translate(point)

    # Customize color if needed
    sphere.paint_uniform_color([1, 0, 0])  # Red color
    return sphere

def draw_apriltag(image, apriltag):
    (ptA, ptB, ptC, ptD) = apriltag.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # draw the bounding box of the AprilTag detection
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(apriltag.center[0]), int(apriltag.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    # draw the tag family on the image
    tag_id = apriltag.tag_id
    cv2.putText(image, str(tag_id), (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
def draw_apriltag_with_uinits(image, apriltag, x_world, y_world):
    (ptA, ptB, ptC, ptD) = apriltag.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # draw the bounding box of the AprilTag detection
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(apriltag.center[0]), int(apriltag.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    # draw the tag family on the image
    tag_id = apriltag.tag_id
    cv2.putText(image, str("{0:.1f},{1:.1f}".format(x_world, y_world)), (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
def draw_text(image, text, x_raw, y_raw):
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # White color in BGR format
    line_type = cv2.LINE_AA

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=1)

    # Calculate the position of the text
    x = int((x_raw - text_size[0]) / 2)
    y = int((y_raw + text_size[1]) / 2)

    # Draw the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness=2, lineType=line_type)