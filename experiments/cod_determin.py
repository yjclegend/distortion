import numpy as np
import matplotlib.pyplot as plt



from common.straightline import straightness
from common.Homography import fundamental, left_epipole

from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid, gen_norm_line


camera = CameraDistortion(2000, 6, 0.005)

def hartley():
    pattern = gen_grid(camera.get_fov(extend=1.5), 50)
    image_d = camera.distort(pattern, k1=-0.01, d=1, theta=np.radians(5/60))
    index = camera.valid_range(image_d)
    pattern = pattern[index]
    image_d = image_d[index]

    pattern = camera.distort(pattern, theta=np.radians(15))
    f = fundamental(image_d[:-200], pattern[:-200])

    cod = left_epipole(f)
    print(cod)

def vera():
    cs = np.linspace(-0.0005, 0.0005, 100)
    min_s, minc = 10, 10
    for c in cs:
        pattern = gen_norm_line(np.pi/2, camera.get_fov(), c, crop=False)
        image_d = camera.distort(pattern, k1=-0.01, d=2, theta=np.radians(5/60))
        s = straightness(image_d)
        if s < min_s:
            min_s = s
            minc = c
    print(minc)

def wang():
    pass

# hartley()
vera()