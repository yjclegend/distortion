import pickle

import numpy as np

from common.geometry import distancePoint

class DistortionCorrection:

    def __init__(self, name='distortion_correction'):
        self.name = name
    
    def set_degree(self, deg):
        pass
    
    def estimate(self, source, target, **kwargs):# **kwargs may include cod
        pass

    def undistort(self, points):
        pass

    def evaluate(self, source, dest):
        pred = self.undistort(source)
        mse = np.mean(distancePoint(dest, pred))
        return mse

    def save_model(self, name=None):
        filename = self.name
        if name is not None:
            filename += name
        f = open(f"saved_model/{filename}.pkl", 'wb')
        pickle.dump(self, f)
    
    def save_to_path(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)

    def load_model(name):
        f = open(f"saved_model/{name}.pkl", 'rb')
        return pickle.load(f)

    def load_path(path):
        f = open(path,'rb')
        return pickle.load(f)



