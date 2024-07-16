import cv2, os
import numpy as np
import matplotlib.pyplot as plt

from common.zebra import LineSegment

def scene_lines(path, group=False):
    files = os.listdir(path)
    lines = list()
    for f in files:
        image = cv2.imread(os.path.join(path, f), 0)
        if group:
            lines.append(findlines(image))
        else:
            lines.extend(findlines(image))
    return lines

def findlines(image):
    _, thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = list()
    plt.imshow(image)
    for c in contours:
        if c.shape[0] < 4000:
            continue
        sample = np.reshape(c, (c.shape[0], 2))
        x_range = np.max(sample[:, 0]) - np.min(sample[:, 0])
        y_range = np.max(sample[:, 1]) - np.min(sample[:, 1])
        col = 0
        if x_range < y_range:
            col = 1
        sample = sample[sample[:, col].argsort()]
        line = LineSegment(sample)
        lines.append(line.smoothed)
        plt.scatter(line.smoothed[:, 0], line.smoothed[:, 1])
    plt.show()
    return lines

def test1():
    image = cv2.imread("data/ceil.bmp", 0)
    findlines(image)

if __name__ == "__main__":
    lines = scene_lines("data/scene202303221500")
    
    for line in lines:
        plt.scatter(line[:, 0], line[:, 1])
    plt.show()
    # test1()