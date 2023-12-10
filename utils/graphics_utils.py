import cv2

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