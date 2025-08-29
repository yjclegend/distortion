from PIL import Image
import numpy as np
import os

from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import U_RFM_DC


de1:U_RFM_DC = DistortionCorrection.load_model('de_laptop_reverse')


# file = 'doc/journal/decoupled/figures/nse_compare/chessboard%d.jpg' % i
file = 'data/decoupled/laptop/outdoor/20241104-120645-936.jpg'

image = np.array(Image.open(file))
corrected = de1.remap_inverse(image)
# import matplotlib.pyplot as plt
# plt.imshow(corrected)
# plt.show()
Image.fromarray(corrected).save('image_output_inverse.png')