# simulate the direction of tangential distortion

import numpy as np
import matplotlib.pyplot as plt
from common.geometry import distancePoint
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid, gen_ring

camera = CameraDistortion(2000, 6, 0.01)
grid = gen_grid(camera.get_fov(), 20)
ring = gen_ring(camera.get_fov()/2, 20)


def plot_direction(pattern):
    image_u = camera.distortion_free(ring)
    image_drt = camera.distort(pattern, k1=-0.01, theta=np.radians(5))
    image_dr = camera.distort(pattern, k1=-0.01)
    distance = distancePoint(image_dr, image_drt)
    vect = image_drt - image_dr

    fig1, ax1 = plt.subplots()
    M = np.hypot(vect[:, 0], vect[:, 1])
    Q = plt.quiver(image_dr[:, 0], image_dr[:, 1], vect[:, 0], vect[:, 1], angles='xy', scale=1, scale_units='xy', units='inches', width=0.015, color='green')
    # qk = ax1.quiverkey(Q, 0.5, 0.65, 256, 'tangential shift', labelpos='N')
    ax1.scatter(image_dr[:, 0], image_dr[:, 1], s=4, label='radial')
    ax1.scatter(image_drt[:, 0], image_drt[:, 1], s=4, label='radial+tangential')
    ax1.scatter([0], [0], color='red', s=4, label='COD')
    ax1.legend(loc=3)
    ax1.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_direction(pattern=ring)