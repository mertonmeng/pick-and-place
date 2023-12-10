import numpy as np
import math

# TOP_LEFT_TAG_ID = 183
# TOP_RIGHT_TAG_ID = 182
# BOTTOM_LEFT_TAG_ID = 159
# BOTTOM_RIGHT_TAG_ID = 158

TOP_LEFT_TAG_ID = 14
TOP_RIGHT_TAG_ID = 15
BOTTOM_LEFT_TAG_ID = 38
BOTTOM_RIGHT_TAG_ID = 39
TAG_LIST = [TOP_LEFT_TAG_ID, TOP_RIGHT_TAG_ID, BOTTOM_LEFT_TAG_ID, BOTTOM_RIGHT_TAG_ID]

TARGET_BOTTOM_LEFT_TAG_ID = 62
TARGET_BOTTOM_RIGHT_TAG_ID = 63
TARGET_TAG_LIST = [TARGET_BOTTOM_LEFT_TAG_ID, TARGET_BOTTOM_RIGHT_TAG_ID]

def get_april_tags_2d_coords(april_tags):
    x_list = []
    y_list = []
    point_dict = {}
    for apriltag in april_tags:
        x_list.append(apriltag.center[0])
        y_list.append(apriltag.center[1])
        point_dict[apriltag.tag_id] = apriltag.center

    theta_1 = np.arctan2(point_dict[TOP_LEFT_TAG_ID][1] - point_dict[BOTTOM_LEFT_TAG_ID][1], point_dict[TOP_LEFT_TAG_ID][0] - point_dict[BOTTOM_LEFT_TAG_ID][0])
    theta_1 = math.degrees(theta_1)
    theta_2 = np.arctan2(point_dict[TOP_RIGHT_TAG_ID][1] - point_dict[BOTTOM_RIGHT_TAG_ID][1], point_dict[TOP_RIGHT_TAG_ID][0] - point_dict[BOTTOM_RIGHT_TAG_ID][0])
    theta_2 = math.degrees(theta_2)
    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    return (x_mean, y_mean, (theta_1 + theta_2) / 2)

def get_april_tags_2d_coords_gripper(april_tags):
    x_list = []
    y_list = []
    point_dict = {}
    for apriltag in april_tags:
        if apriltag.tag_id not in TAG_LIST:
            continue
        x_list.append(apriltag.center[0])
        y_list.append(apriltag.center[1])
        point_dict[apriltag.tag_id] = apriltag.center

    if len(point_dict) != 4:
        return None

    theta_1 = np.arctan2(point_dict[TOP_LEFT_TAG_ID][1] - point_dict[BOTTOM_LEFT_TAG_ID][1], point_dict[TOP_LEFT_TAG_ID][0] - point_dict[BOTTOM_LEFT_TAG_ID][0])
    theta_2 = np.arctan2(point_dict[TOP_RIGHT_TAG_ID][1] - point_dict[BOTTOM_RIGHT_TAG_ID][1], point_dict[TOP_RIGHT_TAG_ID][0] - point_dict[BOTTOM_RIGHT_TAG_ID][0])
    
    theta_1 = theta_1 if theta_1 < 0 else theta_1 - 2*math.pi
    theta_2 = theta_2 if theta_2 < 0 else theta_2 - 2*math.pi

    theta_1_deg = math.degrees(theta_1)
    theta_2_deg = math.degrees(theta_2)

    dist1 = math.dist(point_dict[TOP_LEFT_TAG_ID], point_dict[BOTTOM_LEFT_TAG_ID])
    dist2 = math.dist(point_dict[TOP_RIGHT_TAG_ID], point_dict[BOTTOM_RIGHT_TAG_ID])
    dist = (dist1 + dist2) / 2
    x_mean = np.mean(x_list) + math.cos((theta_1 + theta_2) / 2) * dist * 2.5
    y_mean = np.mean(y_list) + math.sin((theta_1 + theta_2) / 2) * dist * 2.5
    
    return (x_mean, y_mean, (theta_1_deg + theta_2_deg) / 2)

def get_april_tags_2d_coords_target(april_tags):
    x_list = []
    y_list = []
    point_dict = {}
    for apriltag in april_tags:
        if apriltag.tag_id not in TARGET_TAG_LIST:
            continue
        x_list.append(apriltag.center[0])
        y_list.append(apriltag.center[1])
        point_dict[apriltag.tag_id] = apriltag.center

    if len(point_dict) != 2:
        return None

    theta_1 = np.arctan2(point_dict[TARGET_BOTTOM_RIGHT_TAG_ID][1] - point_dict[TARGET_BOTTOM_LEFT_TAG_ID][1], point_dict[TARGET_BOTTOM_RIGHT_TAG_ID][0] - point_dict[TARGET_BOTTOM_LEFT_TAG_ID][0])
    theta_1 = theta_1 if theta_1 < 0 else theta_1 - 2*math.pi
    theta_1_deg = math.degrees(theta_1)

    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    
    return (x_mean, y_mean, theta_1_deg)