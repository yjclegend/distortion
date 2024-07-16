from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import MetricDMDC, MetricPolyDC, MetricRCPMDC, MetricRFMDC, MetricRTMDC
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid

class DC_TestBeach:
    def __init__(self, ):
        self.models:dict[str, DistortionCorrection] = {
            'rtm': MetricRTMDC(),
            'dm' : MetricDMDC(),
            'pm' : MetricPolyDC(),
            'rfm' : MetricRFMDC(),
            # 'rcpm' : MetricRCPMDC()
        }
        self.camera = CameraDistortion(2000, 6, 0.01)
    
    def gen_data(self,k=-0.01, theta=0, noise=0, gs=100):

        pattern = gen_grid(self.camera.range_i, gs)
        self.image_u = self.camera.distortion_free(pattern)
        self.image_d = self.camera.distort(pattern, k1=k, theta=np.radians(theta), noise=noise)
        # self.image_u *= 3
    
    def smoke(self):
        for name, model in self.models.items():
            model.estimate(self.image_d, self.image_u)
            mse = model.evaluate(self.image_d, self.image_u)
            print(f'{name}: {mse}')
    
    def case_cod(self):
        self.gen_data(k=-0.05)
        shift = np.linspace(1, 40, 100)
        for name, model in self.models.items():
            mse_list = list()
            for s in shift:
                cod = (s, s)
                model.estimate(self.image_d, self.image_u, cod=cod)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(shift, mse_list, label=name)
        plt.legend()
        plt.xlabel('cod shift')
        plt.ylabel('E(pixels)')
        plt.show()

    def case_k(self):
        # ks = np.linspace(-0.06, -0.00, 20)
        ks = np.linspace(0, 0.16, 20)
        for name, model in self.models.items():
            mse_list = list()
            for k in ks:
                self.gen_data(k=k)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(np.abs(ks), mse_list, label=name, marker='.')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('E(pixels)')
        plt.show()
    
    def case_tan(self):
        tans = np.linspace(0, 12, 20)/60
        tan_amount = np.linspace(0, 5, 100)
        for name, model in self.models.items():
            mse_list = list()
            for tan in tans:
                self.gen_data(k=0.05, theta=tan)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(tans, mse_list, label=name, marker='.')
        plt.legend()
        plt.xlabel('θ(°)')
        plt.ylabel('E(pixels)')
        plt.show()

    def case_noise(self):
        noise = np.linspace(0, 1, 20)
        for name, model in self.models.items():
            mse_list = list()
            for n in noise:
                self.gen_data(noise=n, theta=0.1)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(noise, mse_list, label=name, marker='.')
        plt.legend()
        plt.xlabel('noiseR level')
        plt.ylabel('E(pixels)')
        plt.show()

    def case_degree(self):
        degrees = np.arange(3, 12, 2)
        self.gen_data(k=0.03,theta=0.0)
        for name, model in self.models.items():
            mse_list = list()
            for deg in degrees:
                model.set_degree(deg)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(degrees, mse_list, label=name, marker='.')
        plt.legend()
        plt.xticks(degrees)
        plt.xlabel('degree')
        plt.ylabel('E(pixels)')
        plt.show()
    
    def case_sparsity(self):
        gs = 100
        self.gen_data(k=0.05,theta=0.1, gs=100)
        # ps = range(1,4)
        ratio = np.linspace(0.001, 0.2, 100)
        for name, model in self.models.items():
            mse_list = list()
            for r in ratio:
                X_train, X_test, y_train, y_test = train_test_split(self.image_d, self.image_u, test_size=1-r, random_state=42)
                model.estimate(X_train, y_train)
                mse = model.evaluate(self.image_d, self.image_u)
                # mse = model.evaluate(X_test, y_test)
                mse_list.append(mse)
            plt.plot(ratio, mse_list, label=name, marker='.')
        plt.legend()
        plt.xlabel('ratio')
        plt.ylabel('E(pixels)')
        plt.show()
    
    def case_bias(self):
        self.gen_data(k=0.05,theta=0.0, noise=0.8)
        ratios = np.linspace(0.2, 0.8, 100)
        for name, model in self.models.items():
            mse_list = list()
            for r in ratios:
                train_idx = np.where(self.image_d[:, 0] < 1000*r)
                test_idx = np.where(self.image_d[:, 0] > 1000*r)
                train_x = self.image_d[train_idx]
                train_y = self.image_u[train_idx]

                model.estimate(train_x, train_y)
                mse = model.evaluate(self.image_d[test_idx], self.image_u[test_idx])
                # mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)

            plt.plot(ratios, mse_list, label=name)
        plt.legend()
        plt.xlabel('bias ratio')
        plt.ylabel('E(pixels)')
        plt.show()


if __name__ == "__main__":
    tb = DC_TestBeach()
    # tb.case_cod()
    # tb.case_degree()
    # tb.case_k()
    # tb.case_tan()
    tb.case_noise()
    
    # tb.case_sparsity()
    # tb.case_bias()