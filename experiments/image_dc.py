
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.calibration.zhangOpenCV import ZhangOpenCV
from distortion_correction.metric.regression_dc import U_RFM_DC




zc:ZhangOpenCV = ZhangOpenCV.load_model('zc_laptop')
de1:U_RFM_DC = DistortionCorrection.load_model('de_laptop')
de2:U_RFM_DC = DistortionCorrection.load_model('de_laptop_c1')

import os 
import cv2

path = 'data/decoupled/laptop/outdoor'
files = os.listdir(path)
count = 1
for file in files:
    image = cv2.imread(os.path.join(path, file))
    corrected = de1.remap(image)

    cv2.imwrite(f'%d.png' % count, corrected)
    count += 1
