from common.plot import my_plot_style
from common.straightline import line_from_grid, straightness
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.calibration.zhangOpenCV import ZhangOpenCV
from distortion_correction.metric.regression_dc import U_RFM_DC

import matplotlib.pyplot as plt
import cv2, os

def plot_straightness(lines):
    nse_zc = []
    nse_dl = []
    nse_de1 = []
    nse_de2 = []
    for line in lines:
        zc_cor = zc.undistort(line)
        dl_cor = dl.undistort(line)
        de_cor1 = de1.undistort(line)
        de_cor2 = de2.undistort(line)
        s_zc = straightness(zc_cor)
        s_dl = straightness(dl_cor)
        s_de1 = straightness(de_cor1)
        s_de2 = straightness(de_cor2)
        nse_zc.append(s_zc)
        nse_dl.append(s_dl)
        nse_de1.append(s_de1)
        nse_de2.append(s_de2)


    # plt.scatter(line[:, 0], line[:, 1], color='red')
    #     plt.plot(de_cor[:, 0], de_cor[:, 1], color='blue')
    # plt.show()
    plt.plot(nse_zc, label='calibration', marker='o', linestyle='', linewidth=1.5, markersize=3)
    plt.plot(nse_dl, label='RDTR', marker='*', linestyle='', linewidth=1.5, markersize=3)
    plt.plot(nse_de1, label='decoupled c2', marker='s', linestyle='', linewidth=1.5, markersize=3)
    plt.plot(nse_de1, label='decoupled c1', marker='^', linestyle='', linewidth=1.5, markersize=3)

    plt.legend()
    plt.xlabel('line')
    plt.ylabel('NSE')
    my_plot_style()
    plt.show()


zc:ZhangOpenCV = ZhangOpenCV.load_model('zc_laptop')
dl:ZhangOpenCV = ZhangOpenCV.load_model('zc_dl')
de1:U_RFM_DC = DistortionCorrection.load_model('de_laptop')
de2:U_RFM_DC = DistortionCorrection.load_model('de_laptop_c1')

# file = 'data/decoupled/laptop/46x30/20241104-102151-889.jpg'
# file = 'data/decoupled/laptop/46x30/20241104-110355-475.jpg'
# file = 'data/decoupled/laptop/46x30/20241104-110358-908.jpg'
# file = 'data/decoupled/laptop/46x30/20241104-110400-071.jpg'
# file = 'data/decoupled/laptop/46x30/20241104-110403-633.jpg'
for i in range(1, 6):
# for i in range(1, 4):
    file = 'doc/journal/decoupled/figures/nse_compare/chessboard%d.jpg' % i
    # file = 'doc/journal/decoupled/figures/outdoor/outdoor%d.jpg' % i
    base, ext = os.path.splitext(file)
    save_path = f"{base}_u.png"
    hori, vert = line_from_grid(file, size=(46, 30))
    plot_straightness(vert)
    plot_straightness(hori)

    # image = cv2.imread(file)
    # corrected = de1.remap(image)
    # cv2.imwrite(save_path, corrected)
# file = 'doc/journal/decoupled/figures/nse_compare/chessboard1.jpg'

# hori, vert = line_from_grid('data/decoupled/glass_30x46center.bmp', size=(30, 46))
# hori, vert = line_from_grid(file, size=(46, 30))


# image = cv2.imread(file)

# for l in hori:

#     plt.plot(l[:, 0], l[:, 1], color='red')

# for l in vert:
#     plt.plot(l[:, 0], l[:, 1], color='blue')

# corrected = de1.remap(image)
# plt.imshow(corrected, cmap='gray')
# plt.show()
# Save the image using OpenCV
# cv2.imwrite('image_output.png', corrected)
# plot_straightness(vert)
# plot_straightness(hori)