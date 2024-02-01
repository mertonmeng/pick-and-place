import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
    image_count = 0
    while True:

        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()

        if not ret:
            continue
            
        # Plot the image
        cv2.imshow("Color Image",color_image)
        
        # Press q key to stop
        retval = cv2.waitKey(1)
        if retval == ord('q'): 
            break
        elif retval == ord('w'):
            cv2.imwrite(f"{image_count}.png", color_image)
            image_count += 1
		