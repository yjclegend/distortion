from matplotlib import pyplot as plt
import numpy as np
from distortion_correction.metric.regression_dc import U_DM_N, MetricDMDC, U_PM, MetricRCPMDC, U_RFM_DC, MetricRTMDC, MetricRTMDC1, U_RTM_N, MetricRadialPolyDC
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid



camera = CameraDistortion(2000, 12, 0.01)
pattern = gen_grid(camera.range_i, 100)
pattern = gen_grid(camera.get_fov(extend=1.5), 100)
image_u = camera.distortion_free(pattern)
image_d = camera.distort(pattern, k1=0.1, theta=np.radians(5/60))
# image_u *= 3
# image_u[:, 0] += 10
# image_u[:, 1] += 5

# image_d[:, 0] += 1000
# image_d[:, 1] += 1000

# cod=(1000, 1000)
# plt.scatter(image_u[:, 0], image_u[:, 1])
# plt.scatter(image_d[:, 0], image_d[:, 1])
# plt.show()

def testPM():
    dm = U_PM(11)
    dm.estimate(image_d, image_u)
    print(dm.model.intercept_)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    image_c = dm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def testRFM():
    rfm = U_RFM_DC(7)
    rfm.estimate(image_d, image_u)
    mse = rfm.evaluate(image_d, image_u)
    print(mse)

    image_c = rfm.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()


def testRTM():
    cod = (0, 0)
    dm = MetricRTMDC1(5)
    # dm = MetricRTMDC(5)
    dm.estimate(image_d, image_u, cod=cod)
    print(dm.model.intercept_)
    print(dm.model.coef_)
    mse = dm.evaluate(image_d, image_u)
    print(mse)

    # image_c = dm.undistort(image_d)

    # plt.scatter(image_u[:, 0], image_u[:, 1])
    # plt.scatter(image_c[:, 0], image_c[:, 1])
    # plt.show()

def test_U_RTM_N():
    m = U_RTM_N(5, tangential=True, fixk1=False)
    # m = U_RTM_N(5)
    m.estimate(image_d, image_u)
    m.model_summary()
    print("cod: ", m.params[:2])
    mse = m.evaluate(image_d, image_u)
    print(mse)
    image_c = m.undistort(image_d)

    plt.scatter(image_u[:, 0], image_u[:, 1])
    plt.scatter(image_c[:, 0], image_c[:, 1])
    plt.show()

def test_U_DM_N():
    m = U_DM_N(5, fixk1=False)
    # m = U_RTM_N(5)
    m.estimate(image_d, image_u)
    print(m.params)
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

if __name__ == "__main__":
    # testPM()
    # testRFM()
    # testRTM()
    # test_U_RTM_N()
    test_U_DM_N()
    # testDM()
    # testRCPM()
    # above done
    # testRPM()
    # compare()
    pass
    