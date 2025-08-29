from PIL import Image
import numpy as np
import os

from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import U_RFM_DC


de1:U_RFM_DC = DistortionCorrection.load_model('de_laptop_reverse')

# file = 'data/decoupled/laptop/46x30/20241104-102151-889.jpg'
# for i in range(1, 6):
for i in range(1, 4):
    # file = 'doc/journal/decoupled/figures/nse_compare/chessboard%d.jpg' % i
    file = 'doc/journal/decoupled/figures/outdoor/outdoor%d.jpg' % i
    base, ext = os.path.splitext(file)
    save_path = f"{base}_ub.png"
    image = np.array(Image.open(file))
    corrected = de1.remap_inverse(image)
    # import matplotlib.pyplot as plt
    # plt.imshow(corrected)
    # plt.show()
    Image.fromarray(corrected).save(save_path)