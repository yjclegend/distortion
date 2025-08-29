import pickle

import numpy as np

from common.geometry import distancePoint

class DistortionCorrection:

    def __init__(self, name='distortion_correction'):
        self.name = name
        self.homo = None
    
    def set_degree(self, deg):
        pass
    
    def estimate(self, source, target, **kwargs):# **kwargs may include cod
        from common.homography import homography
        # a rough homography transform plane coordinates to pixel coorinates
        self.homo = homography(target, source)

    def undistort(self, points):
        pass

    def remap(self, image):
        height, width = image.shape[:2]
        dist_free_img = np.zeros((height, width, 3), dtype=np.uint8)

        x_coords = np.arange(width)
        y_coords = np.arange(height)
        original_points = np.array(np.meshgrid(x_coords, y_coords)).transpose(1, 2, 0)  # Shape (height, width, 2)
        original_points = original_points.reshape(-1, 2)  # Flatten to (height * width, 2)
        center_x, center_y = width//2, height//2
        center_idx = center_y * width + center_x
        # import time
        # s1 = time.time()
        corrected = self.undistort(original_points)
        # s2 = time.time()
        # duration = s2-s1
        # print("duration: ", duration)
        # Convert to homogeneous coordinates by adding a third column of ones
        corrected = np.hstack((corrected, np.ones((corrected.shape[0], 1))))
        # x_min, x_max = corrected[:, 0].min(), corrected[:, 0].max()
        # y_min, y_max = corrected[:, 1].min(), corrected[:, 1].max()
        # print(f"x: {x_min} ~ {x_max}")
        # print(f"y: {y_min} ~ {y_max}")
        # exit()
        scale = np.array([[0.45, 0, 0],
                     [0, 0.45, 0], 
                     [0, 0, 1]])
        shrinked = scale @ self.homo
        corrected = corrected @ shrinked.T
        # Convert from homogeneous coordinates back to 2D by dividing by the third coordinate
        corrected = corrected[:, :2] / corrected[:, 2, np.newaxis]
        corrected_center = corrected[center_idx]
        shift_x = center_x - corrected_center[0]
        shift_y = center_y - corrected_center[1]
        corrected[:, 0] += shift_x
        corrected[:, 1] += shift_y

        for i in range(width):
            for j in range(height):
                idx = j * width + i
                ux, uy = corrected[idx]
                if ux >= 0 and ux < width and uy >= 0 and uy < height:
                    dist_free_img[int(uy), int(ux)] = image[j, i]
        return dist_free_img

    def remap_inverse(self, image):
        from scipy.ndimage import map_coordinates
        # -45,82
        #-23, 52
        df_x = np.arange(-45, 82, 0.06)
        df_y = np.arange(-23, 52, 0.06)[::-1]
        coords = np.array(np.meshgrid(df_x, df_y)).T.reshape(-1, 2)
        xx, yy = np.meshgrid(df_x, df_y, indexing='xy')
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)


        W = len(df_x)
        H = len(df_y)
        print("W, H:", W, H)
        imageH, imageW = image.shape[:2]
        print("imageH, imageW:", imageH, imageW)
        # exit()
        color_array = np.zeros((H * W, 3), dtype=np.uint8)  # shape: (num_pixels, 3)

        # coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])


        # height, width = image.shape[:2]
        # dist_free_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
        # homo_inv = np.linalg.inv(self.homo)
        # mapped = coords_h @ self.homo.T
        # mapped_2d = mapped[:, :2] / mapped[:, 2, np.newaxis]
        mapped_2d = coords
        # mapped_2d now contains the coordinates on the calibration target plane
        distorted = self.undistort(mapped_2d)
        # Prepare coordinates for map_coordinates: shape should be (2, N), with order [y, x]
        coords_yx = [distorted[:, 1], distorted[:, 0]]  # [y, x] order

        for c in range(3):
            color_array[:, c] = map_coordinates(
                image[..., c], coords_yx, order=3, mode='constant'
            )
        # for i in range(len(color_array)):
        #     dist_x, dist_y = int(distorted[i, 0]), int(distorted[i, 1])
        #     if 0 <= dist_x < imageW and 0 <= dist_y < imageH:
        #         color_array[i] = image[dist_y, dist_x]
        # out = np.stack(channels, axis=-1)  # shape (N, 3)
        # out = out.reshape((height, width, C))
        # return out
        return color_array.reshape((H, W, 3))
        


    def evaluate(self, source, dest):
        pred = self.undistort(source)
        mse = np.mean(distancePoint(dest, pred))
        return mse

    def save_model(self, name=None):
        filename = self.name
        if name is not None:
            filename = name
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



