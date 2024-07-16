"""
the distortion free image on the unit plane uses millimeter units
the unit plane has z=1 in camera coordinate
"""

import numpy as np
import matplotlib.pyplot as plt

from common.geometry import distancePoint
from simulation.gen_sample import gen_grid

class CameraDistortion():
    NUM = 100
    def __init__(self, w, f, ps):
        self.w = w
        self.f = f
        self.ps = ps
        self.fp = f / ps
        self.range_i = self.w * self.ps / self.f
    
    def get_fov(self, extend=1):
        return self.range_i * extend
    
    # def set_image_i(self, image_i):
    #     self.image_i = image_i

    # def gen_image_u(self):
    #     self.image_u = np.ones_like(self.image_i)
    #     self.image_u[:, :2] = self.image_i[:, :2] * self.fp
    #     self.image_u = self.image_u[:, :2]
    
    def __radial(self, image, k1, division=False):
        x = image[:, 0]
        y = image[:, 1]
        r2 = x**2 + y**2
        ratio = (1 + k1*r2)
        # ratio = (1 + k1*r2 + self.k2 * r2**2 + self.k3 * r2**3)
        if division:
            xr = x / ratio
            yr = y / ratio
        else:
            xr = x * ratio
            yr = y * ratio
        p_radial = np.column_stack((xr, yr, np.zeros_like(xr)))
        return p_radial
    
    # def gen_image_d(self, beta=1, gamma=0, theta=0, d=0, k1=-0.0, noise=0, discretize=False, division=False):
    #     image_a = self.__radial(self.image_i, k1=k1, division=division)
    #     T = np.array((1, 0, 0), (0, 1, -d), (0, 0, 1))
    #     T1 = np.array((1, 0, 0), (0, 1, d), (0, 0, 1))
    #     R = np.array(((1, 0, 0), (0, np.cos(theta), -np.sin(theta)), (0, np.sin(theta), np.cos(theta))))
    #     H = T1*R*T
    #     image_a[:, 2] = 0
    #     image_b = np.dot(H, image_a.T).T
    #     image_b[:, 2] += 1
    #     image_b[:, 0] /= image_b[:, 2]
    #     image_b[:, 1] /= image_b[:, 2]
    #     I = np.array(((self.fp, gamma, 0), (0, self.fp * beta, 0), (0, 0, 1)))
    #     self.image_d = np.dot(I, image_b.T).T
    #     self.image_d = self.image_d[:, :2]
    #     if noise > 0:
    #         noise_mat = np.random.random(self.image_d.shape) * 2 - 1
    #         noise_mat *= noise
    #         self.image_d += noise_mat

    # def truncate(self):
    #     valid = np.where((self.image_d[:, 0]<self.w/2) & (self.image_d[:, 0] > -self.w/2) & (self.image_d[:, 1]<self.w/2) & (self.image_d[:, 1] > -self.w/2))
    #     return valid
    
    def cal_dist(self):
        valid = self.truncate()
        dist = distancePoint(self.image_u , self.image_d)
        dist = dist[valid]
        mean = np.mean(dist)
        max = np.max(dist)
        return mean, max
    
    def distortion_free(self, image_i):
        image_u = np.ones_like(image_i)
        image_u[:, :2] = image_i[:, :2] * self.fp
        image_u = image_u[:, :2]
        return image_u

    def distort(self, image_i, k1=0, theta=0, d=0.00, gamma=0, beta=1, noise=0, division=True):
        image_a = self.__radial(image_i, k1=k1, division=division)
        T = np.array(((1, 0, 0), (0, 1, -d), (0, 0, 1)))
        T1 = np.array(((1, 0, 0), (0, 1, d), (0, 0, 1)))
        R = np.array(((1, 0, 0), (0, np.cos(theta), -np.sin(theta)), (0, np.sin(theta), np.cos(theta))))
        H = np.dot(T1, np.dot(R, T))

        image_a[:, 2] = 0
        image_b = np.dot(H, image_a.T).T
        image_b[:, 2] += 1
        image_b[:, 0] /= image_b[:, 2]
        image_b[:, 1] /= image_b[:, 2]
        I = np.array(((self.fp, gamma, 0), (0, self.fp * beta, 0), (0, 0, 1)))
        image_d = np.dot(I, image_b.T).T
        image_d = image_d[:, :2]
        if noise > 0:
            noise_mat = np.random.random(image_d.shape) * 2 - 1
            noise_mat *= noise
            image_d += noise_mat
        return image_d
    
    def valid_range(self, image_d):
        valid = np.where((image_d[:, 0]<self.w/2) & (image_d[:, 0] > -self.w/2) & (image_d[:, 1]<self.w/2) & (image_d[:, 1] > -self.w/2))
        return valid
    
    def discretize(self, image_d):
        rounded = np.round(image_d)
        dists = distancePoint(rounded, image_d)
        mini = dict()
        index = dict()
        for i in range(len(dists)):
            coor = rounded[i, 0], rounded[i, 1]
            if coor[0] < self.w/2 and coor[0] > -self.w/2 and coor[1] < self.w/2 and coor[1] > -self.w/2:
                if coor not in mini or mini[coor] < dists[i]:
                    mini[coor] = dists[i]
                    index[coor] = i
        return np.array(list(index.values()))



def test_dist():
    ud = CameraDistortion(400, 82, 0.004)
    ud.gen_image_u()
    ud.gen_image_d(theta=np.radians(0.1))
    dist = distancePoint(ud.image_u , ud.image_d)
    mean = np.mean(dist)
    print("mean distortion: ", mean)
    plt.scatter(ud.image_u[:, 0], ud.image_u[:, 1], marker='+')
    plt.scatter(ud.image_d[:, 0], ud.image_d[:, 1], marker='x')
    plt.show()



if __name__ == "__main__":
    test_dist()
    # for fp in range(100, 50000):
    #     for res in range(500, 10000):
    #         ud = Unit2Dist(res, 6, 0.004, fp=fp)
    #         ud.gen_image_u()
    #         ud.gen_image_d(theta=np.radians(0.1))
    #         dist = distancePoints(ud.image_u , ud.image_d)
    #         mean = np.mean(dist)
    # plt.scatter(ud.image_u[:, 0], ud.image_u[:, 1], marker='+')
    # plt.scatter(ud.image_d[:, 0], ud.image_d[:, 1], marker='x')
    # plt.show()
