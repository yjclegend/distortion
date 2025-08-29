import cv2

from distortion_correction.base_model import DistortionCorrection
from distortion_correction.metric.regression_dc import U_RFM_DC

de1:U_RFM_DC = DistortionCorrection.load_model('de_laptop')

file = 'data/decoupled/laptop/outdoor/20241104-123401-542.jpg'
image = cv2.imread(file)
corrected = de1.remap(image)
cv2.imwrite('image_output.png', corrected)