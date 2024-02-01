import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import base64
import json
import requests
from segment_anything import sam_model_registry, SamPredictor

URL = "<VLM_API_URL>"
IMAGE_PATH = "0.png"

DISPLAY_SCALE = 50

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def detect_object(query, img, model=None):
    if model == None:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    else:
        predictor = SamPredictor(model)

    # Converting the image to encoded PNG
    _, png_img = cv2.imencode('.png', img)

    # Converting the image into numpy bytes
    bytes_data = np.array(png_img).tobytes()

    # Converting the bytes to base64 string.
    encoded_str = base64.b64encode(bytes_data).decode("ascii")

    payload = {"query": query, "encoded_image_string": encoded_str, "is_detection": True}

    response = requests.post(URL, json=payload)

    if response.status_code != 200:
        print(response.text)
        return None, None, None

    response_obj = json.loads(response.text)

    if "detection_output" in response_obj:
        detection_output = json.loads(response_obj["detection_output"])
        boxes = detection_output["boxes"]
        # Draw the bounding box

        box = boxes[0]

        print(detection_output)
        input_box = np.array(box)
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask_image = np.uint8(masks[0]) * 255

        contours,_ = cv2.findContours(mask_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # print("Number of contours detected:", len(contours))
        cnt = contours[0]

        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(cnt)
        print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cX = int(np.mean(box[:,0]))
        cY = int(np.mean(box[:,1]))
        
        mid_point_1 = np.int0((box[0] + box[1]) / 2)
        side_length_1 = math.sqrt(np.sum((box[0] - box[1]) ** 2))
        mid_point_2 = np.int0((box[1] + box[2]) / 2)
        side_length_2 = math.sqrt(np.sum((box[1] - box[2]) ** 2))
        mid_point_3 = np.int0((box[2] + box[3]) / 2)
        mid_point_4 = np.int0((box[3] + box[0]) / 2)

        if side_length_1 > side_length_2:
            angle_line = [(mid_point_1[0], mid_point_1[1]), (mid_point_3[0], mid_point_3[1])]
            angle_line_len = side_length_2
        else:
            angle_line = [(mid_point_2[0], mid_point_2[1]), (mid_point_4[0], mid_point_4[1])]
            angle_line_len = side_length_1

        angle = -math.degrees(math.acos((angle_line[0][0] - angle_line[1][0])/angle_line_len)) - 90
        angle = angle if angle >= -180 else angle + 180

        # Drawing the Bounding box and angle line
        annotated_img = cv2.drawContours(img,[box],0,(0,255,255),2)
        annotated_img = cv2.line(annotated_img, angle_line[0], angle_line[1], (0, 255, 0), thickness=2)
        annotated_img = cv2.circle(annotated_img, (cX,cY), 2, (0,255,255), 2)

    return annotated_img, (cX, cY, angle), box, mask_image

if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    query = "Find the charger?"
    annotated_img, pose, box = detect_object(query, img)
    print(pose)
    cv2.imshow("image", annotated_img)
    cv2.waitKey(0)