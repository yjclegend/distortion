from matplotlib import pyplot as plt
import numpy as np
from distortion_correction.metric.regression_dc import MetricDMDC, MetricPolyDC, MetricRCPMDC, MetricRFMDC, MetricRTMDC, MetricRTMDC1, MetricRTMDCN, MetricRadialPolyDC
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid



camera = CameraDistortion(2000, 12, 0.01)
pattern = gen_grid(camera.range_i, 100)
pattern = gen_grid(camera.get_fov(extend=1.5), 100)
image_u = camera.distortion_free(pattern)
image_d = camera.distort(pattern, k1=-0.05, theta=np.radians(5/60), division=False)
# image_u *= 3
# image_u[:, 0] += 0
# image_u[:, 1] += 0

# image_d[:, 0] += 1000
# image_d[:, 1] += 1000
cod = (-5.898387629390528e-05, -22.17879956333988)
# cod=(1000, 1000)
plt.scatter(image_u[:, 0], image_u[:, 1])
plt.scatter(image_d[:, 0], image_d[:, 1])
plt.show()

def testPM():
    dm = MetricPolyDC(11)
    dm.estimate(image_d, image_u)
    print(dm.model.intercept_)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testRFM():
    rfm = MetricRFMDC(7)
    rfm.estimate(image_d, image_u)
    mse = rfm.evaluate(image_d, image_u)
    print(mse)

    image_c = rfm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()


def testRTM():
    dm = MetricRTMDC(5, tangential=True)
    dm.estimate(image_d, image_u, cod=cod)
    print(dm.model.intercept_)
    print(dm.model.coef_)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testRTMNonlinear():
    m = MetricRTMDCN(5)
    m.estimate(image_d, image_u)
    print("cod: ", m.params[:2])
    mse = m.evaluate(image_d, image_u)
    print(mse)
    image_c = m.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testDM():
    dm = MetricDMDC()
    dm.estimate(image_d, image_u, cod=cod)
    mse = dm.evaluate(image_d, image_u)
    print(mse)
    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testRCPM():
    dm = MetricRCPMDC(7)
    dm.estimate(image_d, image_u, cod)
    print(dm.model.intercept_)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testRPM():
    dm = MetricRadialPolyDC(7)
    dm.estimate(image_d, image_u, cod)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()


def compare():
    cod = (-5.898387629390528e-05, -22.17879956333988)
    m1 = MetricRTMDC1(5, tangential=True)
    m2 = MetricRTMDCN(5, tangential=True)
    m1.estimate(image_d, image_u, cod)
    m2.estimate(image_d, image_u)
    print(m1.model.coef_)
    print(m2.params)
    mse1 = m1.evaluate(image_d, image_u)
    mse2 = m2.evaluate(image_d, image_u)
    print(mse1)
    print(mse2)
if __name__ == "__main__":
    # testPM()
    # testRFM()
    # testRTM()
    # testRTMNonlinear()
    # testDM()
    # testRCPM()
    # above done
    # testRPM()
    # compare()
    pass
    