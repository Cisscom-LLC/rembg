import cv2 as cv
import numpy as np


def smooth(content, threshold, sigma_color, sigma_space, return_img = False):
    nparr = np.fromstring(content, np.uint8)
    roi_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    temp_img = roi_img.copy()
    hsv_img = cv.cvtColor(roi_img, cv.COLOR_BGR2HSV)
    # Get the mask for calculating histogram of the object and remove noise
    hsv_mask = cv.inRange(hsv_img, np.array([0., 80., 80.]), np.array([200., 255., 255.]))
    # Make a 3 channel mask
    full_mask = cv.merge((hsv_mask, hsv_mask, hsv_mask))
    # Apply blur on the created image
    blurred_img = cv.bilateralFilter(roi_img, threshold, sigma_color, sigma_space)
    # Apply mask to image
    masked_img = cv.bitwise_and(blurred_img, full_mask)
    # Invert mask
    inverted_mask = cv.bitwise_not(full_mask)
    # Created anti-mask
    masked_img2 = cv.bitwise_and(temp_img, inverted_mask)
    # Add the masked images together
    output_img = cv.add(masked_img2, masked_img)

    if return_img:
        return output_img
    retval, buffer = cv.imencode('.png', output_img)
    return buffer.tobytes()


def grayscale_smooth(content, threshold, sigma_color, sigma_space):
    glam_img = smooth(content, threshold, sigma_color, sigma_space, return_img = True)
    output_img = cv.cvtColor(glam_img, cv.COLOR_BGR2GRAY)
    retval, buffer = cv.imencode('.png', output_img)
    return buffer.tobytes()


def grayscale(content, threshold, sigma_color, sigma_space):
    nparr = np.fromstring(content, np.uint8)
    roi_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    temp_img = roi_img.copy()
    gray_img = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)
    retval, buffer = cv.imencode('.png', gray_img)
    return buffer.tobytes()
