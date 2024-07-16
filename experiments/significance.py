import numpy as np
import matplotlib.pyplot as plt
from common.geometry import distancePoint, distortion_state
from distortion_correction.metric.regression_dc import  MetricRTMDC, MetricRTMDCN

from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid

def distortion_correction(m, image_d, image_u):
    m.estimate(image_d, image_u)
    pred = m.undistort(image_d)
    mse = distancePoint(image_u, pred)
    return np.mean(mse)

RES = 2000
PS = 0.01
FOCAL = 12

ks = np.linspace(0, -0.05, 20)

rmeans = list()
tmeans = list()
rms_rm = list()
rms_rtm = list()

# rtm = MetricRTMDC(5)
# rm = MetricRTMDC(5, tangential=False)
rtm = MetricRTMDCN(7)
rm = MetricRTMDCN(7, tangential=False)

for k in ks:
    camera = CameraDistortion(RES * PS/0.01, FOCAL, 0.01)
    pattern = gen_grid(camera.get_fov(extend=1.5), 100)
    image_u = camera.distortion_free(pattern)
    image_dr = camera.distort(pattern, k1=k, division=False)
    image_dt = camera.distort(pattern, k1=k, theta=np.radians(5/60), division=False)
    index = camera.valid_range(image_dt)
    image_dt = image_dt[index]
    image_u = image_u[index]
    image_dr = image_dr[index]
    meant, _ = distortion_state(image_dr, image_dt)
    tmeans.append(meant)
    meanr, _ = distortion_state(image_dr, image_u)
    rmeans.append(meanr)
    rrm = distortion_correction(rm, image_dt, image_u)
    rms_rm.append(rrm)
    rrtm = distortion_correction(rtm, image_dt, image_u)
    rms_rtm.append(rrtm)


# plt.plot(ps, tmeans, label='mean tangential', marker='+')
print(ks)
print(rmeans)
print("tmeans:", tmeans)
print(rms_rm)
print(rms_rtm)
plt.plot(ks, rms_rm, label='radial model', marker='s', fillstyle='none')
plt.plot(ks, rms_rtm, label='radial+tangential model', marker='^', fillstyle='none')
plt.xlabel('k')
plt.ylabel('E(pixel)')
plt.gca().invert_xaxis()
plt.legend()
plt.show()