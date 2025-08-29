from common.chessboard import prepare_chessboard
from common.homography import homography
from distortion_correction.base_model import DistortionCorrection
from distortion_correction.calibration.zhangOpenCV import ZhangOpenCV

import numpy as np

zc:ZhangOpenCV = ZhangOpenCV.load_model('zc_laptop')
zc:ZhangOpenCV = ZhangOpenCV.load_model('zc_dl')

path = 'data/decoupled/laptop/8x11_2'
objpoints, imgpoints, imageshape = prepare_chessboard(path, (11, 8))
objpoints = objpoints[0][0, :, :2]
print(objpoints)

for points in imgpoints:
    corrected = zc.undistort(points)
    homo = homography(corrected, objpoints)
    corrected = np.hstack((corrected, np.ones((corrected.shape[0], 1))))
    wp = corrected @ homo.T
    wp = wp[:, :2] / wp[:, 2, np.newaxis]
    dists = wp- objpoints
    error = np.mean(dists[:, 0]**2 + dists[:, 1]**2)
    print(error)