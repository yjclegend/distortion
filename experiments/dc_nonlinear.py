import numpy as np
import matplotlib.pyplot as plt
from common.geometry import distancePoint, distortion_state
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import MetricRTMDCN

from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid

def dc(m:DistortionCorrection, source, target,):
    m.estimate(source, target)
    cod = m.params[:2]
    pred = m.undistort(source)
    mse = distancePoint(target, pred)
    # print(np.mean(mse))
    return np.mean(mse), cod

RES = 2000
PS = 0.01
FOCAL = 12

ps = np.linspace(20, 2, 20)
rtm = MetricRTMDCN(5)
rm = MetricRTMDCN(5, tangential=False)
cods = list()
mses = list()
for p in ps:
    p /= 1000
    camera = CameraDistortion(RES*PS/p, FOCAL, p)

    pattern = gen_grid(camera.get_fov(extend=1.5), 100)
    image_u = camera.distortion_free(pattern)
    # image_dr = camera.distort(pattern, k1=-0.05, division=False)
    image_dt = camera.distort(pattern, k1=-0.001, theta=np.radians(5/60), division=False)
    index = camera.valid_range(image_dt)
    image_dt = image_dt[index]
    image_u = image_u[index]
    # image_dr = image_dr[index]

    mse, cod = dc(rtm, image_dt, image_u)
    cods.append(cod)


print(cods)

# cods = np.array(cods)
# print(cods)
# print(mses)
# plt.plot(cods[:, 0])
# plt.plot(cods[:, 1])
# plt.show()