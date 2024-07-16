import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from common.geometry import twoPointCosSin
from scipy.interpolate import UnivariateSpline

import pickle
class LineSegment:
    def __init__(self, sam:np.ndarray, use_poly=True, deg=4):
        self.sample = sam.astype(np.float64)
        self.use_poly = use_poly
        self.degree = deg
        self.rotate()
        self.smooth()
   
    def rotate(self):
        self.costheta, self.sintheta = twoPointCosSin(self.sample[0], self.sample[-1])
        self.rotated = np.dot(self.sample, np.array([[self.costheta, self.sintheta], [-self.sintheta, self.costheta]]).T)
        # self.rotated = np.unique(self.rotated, axis=0)

    def rotateback(self):
        self.smoothed = np.dot(self.smooth_rotated, np.array([[self.costheta, -self.sintheta], [self.sintheta, self.costheta]]).T)
    
    def poly_fit(self):
        fit = np.polyfit(self.rotated[:, 0], self.rotated[:, 1], self.degree)
        self.model = np.poly1d(fit)
    
    def spline_fit(self):
        _, idx = np.unique(self.rotated[:, 0], return_index=True)
        spl = UnivariateSpline(self.rotated[idx, 0], self.rotated[idx, 1])
        spl.set_smoothing_factor(len(self.rotated)//2)
        self.model = spl

    def smooth(self):
        if self.use_poly:
            self.poly_fit()
        else:
            self.spline_fit()
        # x = np.linspace(self.rotated[0, 0], self.rotated[-1, 0], (len(self.rotated)-1) * ratio+1)
        x = np.unique(self.rotated[:, 0])
        y = self.model(x)
        self.smooth_rotated = np.column_stack((x, y))
        # plt.scatter(x, y)
        # plt.show()
        self.rotateback()

def prepare_lines(name):
    path = 'data/' + name
    files = os.listdir(path)
    lines = list()
    smooths = list()
    print(len(files))
    for f in files:
        print(f)
        image = cv2.imread(os.path.join(path, f), 0)
        smooth, line = findlines2(image)
        smooths.append(smooth)
        lines.append(line)
    f = open("lines_" + name[5:] + '.pkl', 'wb')
    pickle.dump(lines, f)
    f = open("smooth_" + name[5:] + '.pkl', 'wb')
    pickle.dump(smooths, f)
        

def findlines1(image):
    # edges = cv2.Canny(image, 100, 200, )
    # edges = cv2.Canny(image, 200, 300, apertureSize=5, L2gradient=True)
    edges = costom_canny(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = list()
    # plt.imshow(image)
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
        # plt.scatter(sample[:, 0], sample[:, 1])
        # line = LineSegment(sample)
        lines.append(sample)
    #     plt.scatter(sample[:, 0], sample[:, 1])
    #     plt.scatter(line.smoothed[:, 0], line.smoothed[:, 1], color='red')
    #     plt.scatter(line2.smoothed[:, 0], line2.smoothed[:, 1], color='blue')
    # plt.show()
    return lines

    

def findlines2(image):
    edges = cv2.Canny(image, 100, 200, apertureSize=3, L2gradient=True)
    edge_points = np.where(edges > 0)
    edge_points = np.column_stack([edge_points[1], edge_points[0]])
    smoothed = list()
    lines = list()
    db = DBSCAN(eps=5, min_samples=2).fit(edge_points)
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    unique_labels = set(labels)
    for k in unique_labels:
        sample = edge_points[labels==k]
        if len(sample) > 1000:
            lines.append(np.array(sample))
            line = LineSegment(sample)
            smoothed.append(line.smoothed)
    return smoothed, lines


def subpixeledge():
    from subpixel_edges import subpixel_edges
    path = 'data/zebra202304030915/'
    files = os.listdir(path)
    image = cv2.imread(os.path.join(path, files[0]), 0)
    edges = subpixel_edges(image, 25, 0, 2)
    plt.imshow(image)
    plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
    plt.show()

import cv2
import numpy as np

import matplotlib.pyplot as plt
def subpixel(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=cv2.FILTER_SCHARR)
    # gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=cv2.FILTER_SCHARR)
    gy = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3).astype('float64')
    gx = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3).astype('float64')
    g = np.sqrt(gx**2 + gy**2)
    edges = np.zeros_like(image)
    THRESH = 100
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            intensity = g[i, j]
            if intensity > THRESH:
                if i == 0 or i == g.shape[0] - 1 or j == 0 or j == g.shape[1] - 1:
                    edges[i, j] == 255
                else:
                    a, b, c = 0, intensity, 0
                    dx, dy = 0, 0
                    if gx[i, j] == 0:
                        dx, dy = 0, np.sign(gy[i, j])
                        a = g[i, j - dy]
                        c = g[i, j + dy]
                    elif gy[i, j] == 0:
                        dx, dy = np.sign(gx[i, j]), 0
                        a = g[i - dx, j]
                        c = g[i + dx, j]
                    elif abs(gx[i, j]) == abs(gy[i, j]):
                        dx, dy = np.sign(gx[i, j]), np.sign(gy[i, j])
                        a = g[i - dx, j - dy]
                        c = g[i + dx, i + dy]
                    elif abs(gx[i, j]) > abs(gy[i, j]):
                        dx = np.sign(gx[i, j])
                    else:
                        pass
                   
    return edges

def costom_canny(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=cv2.FILTER_SCHARR)
    # gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=cv2.FILTER_SCHARR)
    gy = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3).astype('float64')
    gx = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3).astype('float64')
    g = np.sqrt(gx**2 + gy**2)
    plt.imshow(blur)
    # edges = np.zeros_like(image)
    edge_points = list()
    THRESH = 80
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            intensity = g[i, j]
            if intensity > THRESH:
                if i == 0 or i == g.shape[0] - 1 or j == 0 or j == g.shape[1] - 1:
                    # edges[i, j] == 255
                    edge_points.append((i, j))
                else:
                    ax, ay = 0, 0
                    if abs(gx[i, j]) >= abs(gy[i, j]):
                        ax = 1
                    else:
                        ay = 1
                    if gx[i, j] != 0:
                        tan = gy[i, j] / gx[i, j]
                        if abs(tan) > np.tan(np.radians(22.5)) and abs(tan) < np.tan(np.radians(67.5)):
                            ax, ay = 1, int(np.sign(tan))
                    if intensity >= g[i - ax, j - ay] and intensity >= g[i + ax, j + ay]:
                        # edges[i, j] = 255
                        edge_points.append((i, j))
    return edge_points

def test_find_lines():
    path = 'data/zebra202304030915/'
    files = os.listdir(path)
    for f in files[:]:
        image = cv2.imread(os.path.join(path, f), 0)
        edges = cv2.Canny(image, 100, 200, apertureSize=3, L2gradient=True)
        edge_points = np.where(edges > 0)
        edge_points = np.column_stack([edge_points[1], edge_points[0]])
        # db = DBSCAN(eps=30, min_samples=2).fit(edge_points)
        # labels = db.labels_
        
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)
        # unique_labels = set(labels)
        # for k in unique_labels:
        #     line = edge_points[labels==k]
        #     plt.scatter(line[:, 1], line[:, 0])
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        plt.imshow(image)
        plt.scatter(edge_points[:, 0], edge_points[:, 1])
        plt.show()

if __name__ == "__main__":
    subpixeledge()
    # test1()
    # findlines2()
    # test3()
    # test_pixel_level()
    # test_find_lines()
    # prepare_lines('zebra202303220900')
    # load_lines('zebra202304030915', display=True)