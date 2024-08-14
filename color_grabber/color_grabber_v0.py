# %%
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

# %%
import sys
sys.path.append('..')
from semantic_segmentation.semantic_segmentation_v0 import get_segmentation_mask

# %%
SIZE = 256

def grab_colors(img, n_colors=8, weights=(5, 100, 100, 1, 1), resize=False, blur=False):
    if resize:
        img = cv2.resize(img, (SIZE, SIZE))

    if blur:
        # blur image
        for _ in range(blur):
            img = cv2.GaussianBlur(img, (3, 3), 0)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clt = KMeans(n_clusters=n_colors)

    # Get positional row/col channels to append to the points
    # Ex: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
    row_channel = np.arange(img_hsv.shape[0]).reshape(-1, 1)
    row_channel = np.repeat(row_channel, img_hsv.shape[1], axis=1)
    col_channel = np.arange(img_hsv.shape[1]).reshape(1, -1)
    col_channel = np.repeat(col_channel, img_hsv.shape[0], axis=0)
    img_hsv = np.dstack((img_hsv, row_channel, col_channel))

    points = img_hsv.reshape(-1, 5)
    
    # apply mask
    mask = get_segmentation_mask(img)
    mask = mask.reshape(-1)
    points = points[mask.astype(bool)]

    print(f"before: {len(points)}")
    # th = np.percentile(points[:, 0], 25)
    th = np.percentile(points[:, 0], 2)
    points = points[points[:, 0] > th]
    print(f"after: {len(points)}")

    points = points * np.array(weights)
    labels = clt.fit_predict(points)
    cluster_weights = np.bincount(labels)
    cluster_weights = cluster_weights / cluster_weights.sum()
    closest_points = []
    centroids = clt.cluster_centers_
    for centroid in centroids:
        closest_points.append(points[np.argmin(np.linalg.norm(points - centroid, axis=1))])
    closest_points = np.array(closest_points) / np.array(weights)
    centroids = centroids / np.array(weights)
    return centroids, cluster_weights, closest_points
