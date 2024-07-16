"""
everything about the chessboard image processing
"""

import cv2, os
import numpy as np

def grid_ground_truth(r, c):
    """
    generate the ground truth point coordinates for a grid pattern
    """
    grid = np.mgrid[0:r, c:0:-1].T.reshape((r*c, 2))
    return grid

def findchessboard(image=None, filepath=None, size=(11, 8)):
    if filepath is not None:
        image = cv2.imread(filepath, 0)
    if image is None:
        return None
    
    criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    ret, corners = cv2.findChessboardCorners(image, size, cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)#cv2.CALIB_CB_ADAPTIVE_THRESH|
    if ret == True:
        print("corners found")
        corners2 = cv2.cornerSubPix(image, corners, (11,11),(-1,-1), criteria)
        corners2 = np.reshape(corners2, (len(corners2), 2))
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.scatter(corners2[:, 0], corners2[:,1])
        # plt.show()
        return corners2
    else:
        return None

def downsample_corners(corners, shape=(11, 8)):
    idx_list = list()
    for i in range(shape[1]):
        if i % 2 == 0:
            for j in range(shape[0]):
                if j % 2 == 0:
                    idx_list.append(i * shape[0] + j)
    return corners[idx_list]


def create_ground_truth(size=(11, 8)):
    xs_i = np.arange(size[0])
    ys_i = np.arange(size[1])
    x, y = np.meshgrid(xs_i, ys_i)
    grid = np.column_stack([y.flatten(), x.flatten()])
    return grid

def prepare_chessboard(path, size=(11, 8)):
    files = os.listdir(path)
    corners = list()
    objp = np.zeros((1, size[0] * size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for f in files:
        print(f)
        image = cv2.imread(os.path.join(path, f), 0)
        corners = findchessboard(image)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
    import pickle
    with open('data/chessboard20240416.pkl', 'wb') as f:

        pickle.dump(imgpoints, f)
    return objpoints, imgpoints

def findcorners(image=None, filepath=None):
    if filepath is not None:
        image = cv2.imread(filepath, 0)
    if image is None:
        return None


    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype('float32')
    dst = cv2.cornerHarris(gray,2,5,0.04)
    
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.show()



def test_findchessboard():
    img = cv2.imread('data/chessboard_dense/30x46.bmp', 0)
    corners = findchessboard(img, size=(30, 46))
    # corners = downsample_corners(corners, shape=(47, 31))
    # print(corners[:50])
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.show()

def test_findcorners():
    img = cv2.imread('data/chessboard_dense/Image_20230518224319610.bmp')
    corners = findchessboard(img)


def test_dense():
    img = cv2.imread('data/chessboard_dense/Image_20230518224319610.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    white = np.ones_like(color) * 255
    corners = findchessboard(image=gray, size=(47, 31))
    cv2.drawChessboardCorners(color, (47, 31), corners, True)
    print(corners.shape)

    truth = grid_ground_truth(31, 47) * 78
    truth[:, [0, 1]] = truth[:, [1, 0]]

    import matplotlib.pyplot as plt
    # plt.imshow(color)
    plt.imshow(white)
    plt.scatter(corners[:, 0], corners[:, 1], marker='+', label='metric features')
    plt.scatter(truth[:, 0], truth[:, 1], marker='1', label='ground truth')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pass
    create_ground_truth()
    # test_dense()
    # grid_ground_truth(8, 11)
    # test_findchessboard()
