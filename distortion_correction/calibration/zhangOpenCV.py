import numpy as np
import cv2

from distortion_correction.base_model import DistortionCorrection

from common.chessboard import prepare_chessboard

class ZhangOpenCV(DistortionCorrection):
    def __init__(self, shape=(3840, 2748), name='Zhang'):
        super().__init__(name=name)
        self.imgshape=shape
    
    def set_imgshape(self, shape):
        self.imgshape=shape

    def calibrate(self, objpoints, imgpoints):
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.imgshape, None, None)
        print(self.mtx, self.dist)

    def calibrate_k1(self, objpoints, imgpoints):
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.imgshape, None, None,
                        flags=cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_FIX_TANGENT_DIST)
        print(self.mtx, self.dist)
    
    def calibrate_k1k2k3(self, objpoints, imgpoints):
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.imgshape, None, None,
                        flags=cv2.CALIB_RATIONAL_MODEL)
        print(self.mtx, self.dist)

    
    def undistort_image(self, im_path):
        img = cv2.imread(im_path, 0)
        dst = cv2.undistort(img, self.mtx, self.dist, None, None)


    def undistort(self, points):
        dst = cv2.undistortPoints(points, self.mtx, self.dist, P=self.mtx)

        return np.reshape(dst, (len(dst), 2))


def train_zhang(name, path='data/decoupled/glass_calibrate', size=(11, 8)):
    zc = ZhangOpenCV()
    objpoints, imgpoints, imageshape = prepare_chessboard(path, size)
    zc.set_imgshape(imageshape)
    # zc.calibrate(objpoints, imgpoints)
    zc.calibrate_k1(objpoints, imgpoints)

    zc.save_model(name)
    # zc.calibrate_k1(objpoints, imgpoints)
    # zc.save_param('chess202303211245_k1')
    # zc.calibrate_k1k2k3(objpoints, imgpoints)
    # zc.save_param('chess202303211245_k1k2k3')


if __name__ == "__main__":
    # train_zhang('glass_calibrate')
    # train_zhang('zc_laptop', 'data/decoupled/laptop/8x11_glass')
    train_zhang('zc_dl', 'data/decoupled/laptop/46x30')