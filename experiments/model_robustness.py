from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from common.plot import my_plot_style
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import U_DM_N, U_RTM_N, MetricDMDC, U_PM, MetricRCPMDC, U_RFM_DC, MetricRTMDC
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid
DEGREE = 5
# List of different markers to use for each model
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', '*']  # Add as many markers as needed
# Ensure that there are enough markers for each model
marker_cycle = iter(markers)
class DC_TestBench:
    def __init__(self, ):
        self.models:dict[str, DistortionCorrection] = {
            'rtm': U_RTM_N(deg=DEGREE),
            'dm' : U_DM_N(deg=DEGREE),
            'pm' : U_PM(deg=DEGREE),
            'rfm' : U_RFM_DC(deg=DEGREE),
            # 'rcpm' : MetricRCPMDC()
        }

        self.camera = CameraDistortion(2000, 6, 0.01)
    
    def gen_data(self,k=-0.01, theta=0, noise=0, gs=100):

        pattern = gen_grid(self.camera.range_i, gs)
        self.image_u = self.camera.distortion_free(pattern)
        self.image_d = self.camera.distort(pattern, k1=k, theta=np.radians(theta), noise=noise)
        # self.image_u *= 3
    
    def gen_compound(self, k1, k2, k3, p1, p2, gs=100):
        pattern = gen_grid(self.camera.range_i, gs)
        self.image_u = self.camera.distortion_free(pattern)
        self.image_d = self.camera.distort_rtm(pattern, k1, k2, k3, p1, p2)
        

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

        num_experiments = 1
        ks = np.linspace(0, 0.2, 20)
        for name, model in self.models.items():
            mse_means = []
            mse_stds = []
            for k in ks:
                mse_list = []

                # Run multiple experiments for each k value
                for _ in range(num_experiments):
                    self.gen_data(k=k, noise=0.3)
                    model.estimate(self.image_d, self.image_u)
                    mse = model.evaluate(self.image_d, self.image_u)
                    mse_list.append(mse)
                
                # Calculate mean and standard deviation
                mse_mean = np.mean(mse_list)
                # mse_std = np.std(mse_list, ddof=1)  # Sample standard deviation
                
                mse_means.append(mse_mean)
                # mse_stds.append(mse_std)
            # Convert lists to arrays for easier plotting
            mse_means = np.array(mse_means)
            # mse_stds = np.array(mse_stds)
            marker = next(marker_cycle)
            plt.plot(np.abs(ks), mse_means, label=name, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            # Plot the shaded area for the range (mean ± std)
            # plt.fill_between(np.abs(ks), mse_means - mse_stds, mse_means + mse_stds, alpha=0.2)

        plt.legend(loc='upper left')
        plt.xlabel('k')
        plt.ylabel('E(pixels)')


        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid with lighter style
 
        # Tight layout for better spacing
        plt.tight_layout()

        plt.show()
    
    def case_tan(self):
        tans = np.linspace(0, 1, 20)
        tan_amount = np.linspace(0, 5, 100)
        for name, model in self.models.items():
            mse_list = list()
            for tan in tans:
                self.gen_data(k=0.1, theta=tan)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)
            marker = next(marker_cycle)
            plt.plot(tans, mse_list, label=name, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            # plt.plot(tans, mse_list, label=name, marker='.')
        plt.legend()
        plt.xlabel('θ(°)')
        plt.ylabel('E(pixels)')
        my_plot_style()
        plt.show()

    def case_noise(self):
        noise = np.linspace(0, 5, 20)
        for name, model in self.models.items():
            mse_list = list()
            for n in noise:
                self.gen_data(k = 0.1, noise=n, theta=0.1)
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)
            marker = next(marker_cycle)
            plt.plot(noise, mse_list, label=name, marker=marker, markersize=5, linestyle='-', linewidth=1.5)
        plt.legend()
        plt.xlabel('noise level')
        plt.ylabel('E(pixels)')
        my_plot_style()

    def case_degree(self):
        fig, ax1 = plt.subplots(figsize=(3.48, 2.61))
        # Create the second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        import time
        degrees = np.arange(3, 12, 2)
        self.gen_data(k=0.1,theta=0.0, noise=0.3)
        for name, model in self.models.items():
            mse_list = list()
            runtime = list()
            for deg in degrees:
                model.set_degree(deg)
                start = time.time()
                model.estimate(self.image_d, self.image_u)
                end = time.time()
                runtime.append(end-start)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)
            marker = next(marker_cycle)
            ax1.plot(degrees, mse_list, label=name, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            ax2.plot(degrees, runtime, label=name, marker=marker, linestyle='--', linewidth=1.5, markersize=5)
        plt.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid with lighter style
        plt.xticks(degrees)
        ax1.set_xlabel('degree')
        ax1.set_ylabel('E(pixels)')
        ax2.set_ylabel('avg runtime(s)')
        plt.tight_layout
        # my_plot_style()
        plt.show()
    
    def case_sparsity(self):
        # ps = range(1,4)
        ratio = np.linspace(0.001, 0.01, 20)
        num_experiments = 10
        for name, model in self.models.items():
            self.gen_data(k=0.1,theta=0.1, gs=100, noise=0.3)
            mse_means = []
            mins = []
            maxs = []
            
            for r in ratio:
                mse_list = list()
                # Run multiple experiments for each k value
                for _ in range(num_experiments):
                    X_train, X_test, y_train, y_test = train_test_split(self.image_d, self.image_u, test_size=1-r)
                    model.estimate(X_train, y_train)
                    mse = model.evaluate(self.image_d, self.image_u)
                    # mse = model.evaluate(X_test, y_test)
                    mse_list.append(mse)
                # Calculate mean and standard deviation
                mse_mean = np.mean(mse_list)
                # mse_std = np.std(mse_list, ddof=1)  # Sample standard deviation
                mins.append(np.min(mse_list))
                maxs.append(np.max(mse_list))
                mse_means.append(mse_mean)
                # mse_stds.append(mse_std)
            # Convert lists to arrays for easier plotting
            mse_means = np.array(mse_means)
            marker = next(marker_cycle)
            plt.plot(ratio, mse_means, label=name, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            # print(mse_stds)
            # Plot the shaded area for the range (mean ± std)
            plt.fill_between(ratio, mins, maxs, alpha=0.2)
        plt.legend()
        plt.xlabel('train ratio')
        plt.ylabel('E(pixels)')
        my_plot_style()
        plt.show()
    
    def case_bias(self):
        num_experiments = 10
        ratios = np.linspace(0.2, 0.9, 20)
        for name, model in self.models.items():
            # model.set_degree(3)
            mse_means = []
            mins = []
            maxs = []

            for r in ratios:
                mse_list = list()
                for _ in range(num_experiments):
                    self.gen_data(k=0.1,theta=0.1, noise=0.01)
                    max_x = np.max(self.image_d[:, 0])
                    thresh = 2 * max_x * r - max_x
                    train_idx = np.where(self.image_d[:, 0] < thresh)
                    test_idx = np.where(self.image_d[:, 0] > thresh)
                    train_x = self.image_d[train_idx]
                    train_y = self.image_u[train_idx]

                    model.estimate(train_x, train_y)
                # plt.scatter(self.image_d[train_idx, 0], self.image_d[train_idx, 1])
                # plt.scatter(self.image_d[test_idx, 0], self.image_d[test_idx, 1])
                # plt.show()
                # mse = model.evaluate(self.image_d[test_idx], self.image_u[test_idx])
                    mse = model.evaluate(self.image_d, self.image_u)
                    mse_list.append(mse)
                mse_mean = np.mean(mse_list)
                mins.append(np.min(mse_list))
                maxs.append(np.max(mse_list))
                mse_means.append(mse_mean)
            marker = next(marker_cycle)
            plt.plot(ratios, mse_means, label=name, marker=marker, linestyle='-', linewidth=1.5, markersize=5)
            plt.fill_between(ratios, mins, maxs, alpha=0.2)
        plt.legend()
        plt.xlabel('train ratio')
        plt.ylabel('E(pixels)')
        plt.ylim((0, 50))
        my_plot_style()

    def case_compound(self):
        num_experiments = 10
        params = []
        errors = []
        
        for _ in range(num_experiments):
            k1 = np.random.uniform(-0.05, 0.05)
            k2 = np.random.uniform(-0.002, 0.002)
            k3 = np.random.uniform(-0.00001, 0.00001)
            p1 = np.random.uniform(-0.01, 0.01)
            p2 = np.random.uniform(-0.01, 0.01)
            params.append((k1, k2, k3, p1, p2))
            # print(f'k1: {k1}, k2: {k2}, k3: {k3}, p1: {p1}, p2: {p2}')
            self.gen_compound(k1, k2, k3, p1, p2, gs=100)
            # plt.scatter(self.image_u[:, 0], self.image_u[:, 1], s=1)
            # plt.scatter(self.image_d[:, 0], self.image_d[:, 1], s=1)
            # plt.show()
            # continue
            mse_list = []
            for name, model in self.models.items():
                model.estimate(self.image_d, self.image_u)
                mse = model.evaluate(self.image_d, self.image_u)
                mse_list.append(mse)
            errors.append(mse_list)
        # print(params)
        errors = np.log(errors)
        plterrors(errors)
        # plt.boxplot(errors, labels=["rtm", "dm", "pm", "rfm"])
        # plt.ylabel("log(Error)")
        # plt.show()
        # d
def plterrors(errors):
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置样式
    plt.style.use('seaborn-v0_8')  # 使用seaborn样式
    # plt.rcParams['font.family'] = 'Arial'  # 设置字体
    # plt.rcParams['figure.dpi'] = 300      # 提高分辨率

    # 创建图形
    plt.figure(figsize=(4, 3))

    # 绘制箱线图
    box = plt.boxplot(errors, 
                    labels=["RTM", "DM", "PM", "RFM"],
                    patch_artist=True,  # 填充颜色
                    widths=0.6,        # 箱体宽度
                    showfliers=True,   # 显示离群点
                    flierprops=dict(marker='o', markersize=5, 
                                    markerfacecolor='none', markeredgecolor='gray', alpha=0.5))

    # 自定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # 半透明效果

    # 设置中位数线样式
    for median in box['medians']:
        median.set(color='black', linewidth=1.5)

    # 设置标题和标签
    plt.title('Model Performance Comparison', fontsize=12, pad=20)
    plt.ylabel('log(Error)', fontsize=10, labelpad=10)
    plt.xlabel('Models', fontsize=10, labelpad=10)

    # 调整坐标轴
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 移除顶部和右侧边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()     
if __name__ == "__main__":
    tb = DC_TestBench()
    # tb.case_cod()
    # tb.case_degree()
    # tb.case_k()
    # tb.case_tan()
    # tb.case_noise()
    tb.case_compound()
    # tb.case_sparsity()
    # tb.case_bias()