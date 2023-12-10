#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the script used to calibrate camera to a 2D plane with real world coordinate using April tags
It provides 2 calibration model, Neural Network or Linear Regression
"""
import cv2
import numpy as np
from pupil_apriltags import Detector
import torch
import random
import time
import argparse

from torch.optim.lr_scheduler import StepLR
from utils.training_data_utils import split_list
from calibration.models import LinearRegression, Camera2DCalibrationNN
from calibration.models import train_model, evaluate_model

INCH_TO_CM = 2.54

# Linear Regression
def get_linear_regression_model_params():
    learningRate = 0.5
    epochs = 3000

    model = LinearRegression(3, 4)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, 
                    step_size = 100, # Period of learning rate decay
                    gamma = 0.99) # Multiplicative factor of learning rate decay
    return model, criterion, optimizer, scheduler, epochs

def get_neural_network_model_params():
    learningRate = 0.01
    epochs = 3000

    model = Camera2DCalibrationNN(2, 2)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, 
                    step_size = 100, # Period of learning rate decay
                    gamma = 0.95) # Multiplicative factor of learning rate decay
    return model, criterion, optimizer, scheduler, epochs

def draw_points_with_world_coords(image, points, world_coords):
    for i in range(len(points)):
        cX, cY = points[i][0:2]
        x_world, y_world = world_coords[i][0:2]
        image_cx = int(cX * image.shape[1])
        image_cy = int(cY * image.shape[0])

        cv2.circle(image, (image_cx, image_cy), 2, (0, 0, 255), -1)
        # draw the tag family on the image
        cv2.putText(image, str("{0:.1f},{1:.1f}".format(x_world * args.world_unit_normalizer, y_world * args.world_unit_normalizer)), (image_cx, image_cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

def generate_xy_from_apriltag(apriltags, origin_id):
    tag_dict = {}

    origin_row = (origin_id - args.starting_id) // args.id_range_per_row
    origin_col = (origin_id - args.starting_id - origin_row * args.id_range_per_row) % 10

    for tag in apriltags:
        row = (tag.tag_id - args.starting_id) // args.id_range_per_row
        col = (tag.tag_id - args.starting_id  - row * args.id_range_per_row) % 10
        dx_world = (row - origin_row) * INCH_TO_CM
        dy_world = (col - origin_col) * INCH_TO_CM
        tag_dict[tag.tag_id] = [(tag.center[0], tag.center[1]), (dx_world, dy_world)]

    return tag_dict

def generate_2d_linear_calib_dataset(apriltags, dataset, img_width, img_height):
    world_coords_list = []
    image_coords_list = []
    for tag in apriltags:
        x_world, y_world = dataset[tag.tag_id][1]
        x_image, y_image = dataset[tag.tag_id][0]
        world_coords_list.append([x_world / args.world_unit_normalizer, y_world / args.world_unit_normalizer, 1, 1])
        image_coords_list.append([x_image / img_width, y_image / img_height, 1])
        # draw_apriltag_with_uinits(color_image, tag, x_world, y_world)

    image_coords_list_train, image_coords_list_test, world_coords_list_train, world_coords_list_test = split_list(image_coords_list, world_coords_list, 0.9)
    
    image_coords_train = np.array(image_coords_list_train).astype(np.float32)
    world_coords_train = np.array(world_coords_list_train).astype(np.float32)
    image_coords_test = np.array(image_coords_list_test).astype(np.float32)
    world_coords_test = np.array(world_coords_list_test).astype(np.float32)
    return image_coords_train, world_coords_train, image_coords_test, world_coords_test

def generate_2d_nonlinear_calib_dataset(apriltags, dataset, img_width, img_height):
    world_coords_list = []
    image_coords_list = []
    for tag in apriltags:
        x_world, y_world = dataset[tag.tag_id][1]
        x_image, y_image = dataset[tag.tag_id][0]
        world_coords_list.append([x_world / args.world_unit_normalizer, y_world / args.world_unit_normalizer])
        image_coords_list.append([x_image / img_width, y_image / img_height])
        # draw_apriltag_with_uinits(color_image, tag, x_world, y_world)

    image_coords_list_train, image_coords_list_test, world_coords_list_train, world_coords_list_test = split_list(image_coords_list, world_coords_list, 0.9)
    
    image_coords_train = np.array(image_coords_list_train).astype(np.float32)
    world_coords_train = np.array(world_coords_list_train).astype(np.float32)
    image_coords_test = np.array(image_coords_list_test).astype(np.float32)
    world_coords_test = np.array(world_coords_list_test).astype(np.float32)
    return image_coords_train, world_coords_train, image_coords_test, world_coords_test

def draw_axis(image, apriltags, origin_tag_id=90):
    origin, x_pt, y_pt = None, None, None
    for tag in apriltags:
        if tag.tag_id == origin_tag_id:
            origin = (int(tag.center[0]), int(tag.center[1]))
        elif tag.tag_id == origin_tag_id + 1:
            x_pt = (int(tag.center[0]), int(tag.center[1]))
        elif tag.tag_id == origin_tag_id + args.id_range_per_row:
            y_pt = (int(tag.center[0]), int(tag.center[1]))
    
    cv2.arrowedLine(image, origin, x_pt, (255, 0, 0), 2)
    cv2.arrowedLine(image, origin, y_pt, (0, 0, 255), 2)

def prepare_np_random_points(mode):
    points = []
    for i in range(5):
        rand_x = random.random()
        rand_y = random.random()
        if mode == "linear":
            points.append((rand_x, rand_y, 1))
        elif mode == "nn":
            points.append((rand_x, rand_y))
    return np.array(points).astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Model used for calibration", default="nn")
    parser.add_argument("--id_range_per_row", help="Id range of tags per row", default=24)
    parser.add_argument("--num_tags_per_row", help="Number of tags per row", default=10)
    parser.add_argument("--world_unit_normalizer", help="Real World Physical unit", default=20.0)
    parser.add_argument("--starting_id", help="Starting Tag Id", default=14)
    parser.add_argument("--origin_tag_id", help="Origin Tag Id", default=90)

    return parser.parse_args()

if __name__ == "__main__":

    color_image = cv2.imread("calibration/april_tags.png")
    
    args = parse_args()

    at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)

    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    apriltags = at_detector.detect(grayscale_image)

    dataset = generate_xy_from_apriltag(apriltags, 90)
    
    if args.model == "nn":
        image_coords_train, world_coords_train, image_coords_test, world_coords_test = \
            generate_2d_nonlinear_calib_dataset(
                apriltags, 
                dataset, 
                grayscale_image.shape[1], 
                grayscale_image.shape[0])
        model, criterion, optimizer, scheduler, epochs = get_neural_network_model_params()
    elif args.model == "linear":
        image_coords_train, world_coords_train, image_coords_test, world_coords_test = \
            generate_2d_linear_calib_dataset(
                apriltags, 
                dataset, 
                grayscale_image.shape[1], 
                grayscale_image.shape[0])
        model, criterion, optimizer, scheduler, epochs = get_linear_regression_model_params()
    else:
        exit()

    model = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs,
        image_coords_train, 
        world_coords_train, 
        image_coords_test, 
        world_coords_test)
    predicted_val = evaluate_model(model, image_coords_test)
    for i in range(predicted_val.shape[0]):
        print(f"Predicted Coords: {predicted_val[i] * args.world_unit_normalizer}, Actual Coords: {world_coords_test[i] * args.world_unit_normalizer}")

    # Generate few Random points and draw on the image
    rand_points = prepare_np_random_points(args.model)
    predicted_random = evaluate_model(model, rand_points)
    
    draw_points_with_world_coords(color_image, rand_points, predicted_random)

    draw_axis(color_image, apriltags)
    cv2.imshow("Color Image",color_image)
    cv2.imwrite("random_points.png", color_image)
    # Press q key to stop
    while cv2.waitKey(1) != ord('q'): 
        time.sleep(0.1)