import matplotlib.pyplot as plt
import cv2
import numpy as np

# path = '/Users/sdale/repos/find_my_tie/dataset_v0/51fYgoOqgiL._AC_SX679_.jpg'
path = '/Users/sdale/repos/find_my_tie/dataset_v0/515n0PK2+VL._AC_SX679_.jpg'

def get_segmentation_mask(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    binary_image = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise Exception('No contours found')

    # combine contours
    contour = None
    for c in contours:
        if contour is None:
            contour = c
        else:
            contour = np.concatenate((contour, c), axis=0)
    hull = cv2.convexHull(contour)
    # plt.imshow(cv2.drawContours(img, [hull], -1, (255, 0, 0), 2))

    # determine if pixel is inside or outside of hull
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    # plt.imshow(mask, cmap='gray')

    mask = mask // 255

    ### Remove white color
    background_points = img[mask == 0].reshape(-1, 3)
    background_mean = background_points.mean(axis=0)
    ATOL = 10
    new_mask = np.logical_and(
        np.isclose(img[:, :, 0], background_mean[0], atol=ATOL),
        np.logical_and(
            np.isclose(img[:, :, 1], background_mean[1], atol=ATOL),
            np.isclose(img[:, :, 2], background_mean[2], atol=ATOL),
        )
    ).astype(np.uint8)

    return 1-new_mask

def get_segmentation_mask_from_path(path):
    img = cv2.imread(path)
    return get_segmentation_mask(img)