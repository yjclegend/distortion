import numpy as np
import matplotlib.pyplot as plt
from common.geometry import distancePoint, distortion_state
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import  MetricRTMDC

from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid

def dc(m:DistortionCorrection, source, target, cod=(0, 0)):
    m.estimate(source, target, cod=cod)
    pred = m.undistort(source)
    mse = distancePoint(target, pred)
    # print(np.mean(mse))
    return np.mean(mse)

RES = 2000
PS = 0.01
FOCAL = 12

ps = np.linspace(20, 2, 20)
ps = ps[-1:]
rtm = MetricRTMDC(5)
rm = MetricRTMDC(5, tangential=False)
cods = list()
mses = list()
for p in ps:
    p /= 1000
    camera = CameraDistortion(RES*PS/p, FOCAL, p)

    pattern = gen_grid(camera.get_fov(extend=1.5), 100)
    image_u = camera.distortion_free(pattern)
    # image_dr = camera.distort(pattern, k1=-0.05, division=False)
    image_dt = camera.distort(pattern, k1=-0.05, theta=np.radians(5/60), division=False)
    index = camera.valid_range(image_dt)
    image_dt = image_dt[index]
    image_u = image_u[index]
    # image_dr = image_dr[index]

    cxs = np.linspace(-100, 100, 100)
    cys = np.linspace(-100, 100, 100)
    min_mse = 10
    min_cod = (0, 0)
    for cx in cxs:
        for cy in cys:
            mse = dc(rtm, image_dt, image_u, cod=(cx, cy))
            if mse < min_mse:
                min_mse = mse
                min_cod = (cx, cy)
    cods.append(min_cod)
    mses.append(min_mse)

cods = np.array(cods)
print(cods)
print(mses)
plt.plot(cods[:, 0])
plt.plot(cods[:, 1])
plt.show()