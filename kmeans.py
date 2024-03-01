import cv2
import os
import numpy as np
import json
from sklearn.cluster import KMeans
import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image, k=4):
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Define criteria, number of clusters (k) and apply k-means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Count the number of pixels labeled to each centroid
    counts = Counter(labels)

    # Find the most common centroid
    dominant_color = centers[np.argmax(counts)]

    return list(dominant_color)

def get_dominant_color2(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    dominant_color = kmeans.cluster_centers_.astype(int)
    return dominant_color[0]

image = cv2.imread('images/q.jpg')
n_w = 30   
n_h = 15
d_w = image.shape[1] // n_w
d_h = image.shape[0] // n_h
for i in range(n_w):
    for j in range(n_h):
        top_left = [i * d_w, j * d_h]
        bot_righ = [(i+1) * d_w, (j+1) * d_h]
        dom_col = get_dominant_color(image[j*d_h:(j+1)*d_h, i*d_w:(i+1)*d_w], k=20)
        color = (int(dom_col[0]), int(dom_col[1]), int(dom_col[2]))
        cv2.rectangle(image, top_left, bot_righ, color, -1)
        cv2.imshow('grid', image)
        cv2.waitKey(1)
cv2.waitKey(0)

       