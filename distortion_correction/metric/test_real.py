import numpy as np
import matplotlib.pyplot as plt

from common.geometry import distancePoint
from distortion_correction.metric.regression_dc import U_PM, MetricRTMDC

def estimate():
    from common.chessboard import findchessboard, grid_ground_truth
    truth = grid_ground_truth(30, 46)
    corners = findchessboard(filepath='data/chessboard_dense/30x46.bmp', size=(30, 46)).astype('float64')
    assert(truth.shape == corners.shape)

    truth[:, [0, 1]] = truth[:, [1, 0]]
    mgp = U_PM(5)
    mgp.estimate(corners, truth)
    mgp.save_model('30x46_5')

def case1():
    import matplotlib.pyplot as plt
    
    from common.chessboard import findchessboard, grid_ground_truth

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    """
        use 31 * 47 grid points to estimate general polynomial
    """
    truth = grid_ground_truth(31, 47)
    corners = findchessboard(filepath='data/chessboard_dense/23x15.bmp', size=(31, 47)).astype('float64')

    assert(truth.shape == corners.shape)

    truth[:, [0, 1]] = truth[:, [1, 0]]

    X_train, X_test, y_train, y_test = train_test_split(corners, truth, test_size=0.01, random_state=42)
    # X_train = corners[:800]
    # y_train = truth[:800]
    # X_test = corners[800:]
    # y_test = truth[800:]

    mgp = U_PM(5)
    mgp.estimate(X_train, y_train)

    pred_train = mgp.undistort(X_train)
    pred_test = mgp.undistort(X_test)

    # mse_train = mean_squared_error(y_train, pred_train)
    # mse_test = mean_squared_error(y_test, pred_test)

    mse_train = distancePoint(y_train, pred_train)
    mse_test = distancePoint(y_test, pred_test)
    print(mse_train, mse_test)
    plt.scatter(y_train[:, 0], y_train[:, 1])
    plt.scatter(pred_train[:, 0], pred_train[:, 1])
    plt.show()
    plt.scatter(y_test[:, 0], y_test[:, 1])
    plt.scatter(pred_test[:, 0], pred_test[:, 1])
    plt.show()
    
def case2():
    import matplotlib.pyplot as plt
    
    from common.chessboard import findchessboard, grid_ground_truth

    from sklearn.model_selection import train_test_split
    """
        use 31 * 47 grid points to estimate general polynomial
    """
    truth = grid_ground_truth(31, 47)
    corners = findchessboard(filepath='data/chessboard_dense/Image_20230518224958190.bmp', size=(31, 47)).astype('float64')

    assert(truth.shape == corners.shape)

    truth[:, [0, 1]] = truth[:, [1, 0]]

    X_train, X_test, y_train, y_test = train_test_split(corners, truth, test_size=0.1, random_state=42)
    # X_train = corners[:800]
    # y_train = truth[:800]
    # X_test = corners[800:]
    # y_test = truth[800:]

    mgp = U_PM(5)
    mrp = MetricRTMDC(2)
    mgp.estimate(X_train, y_train)
    mrp.estimate(X_train, y_train)

    pred_train_g = mgp.undistort(X_train)
    pred_test_g = mgp.undistort(X_test)
    pred_train_r = mrp.undistort(X_train)
    pred_test_r = mrp.undistort(X_test)

    # mse_train = mean_squared_error(y_train, pred_train)
    # mse_test = mean_squared_error(y_test, pred_test)

    mse_train_g = distancePoint(y_train, pred_train_g)
    mse_test_g = distancePoint(y_test, pred_test_g)
    print(np.mean(mse_train_g), np.mean(mse_test_g))

    mse_train_g = distancePoint(y_train, pred_train_r)
    mse_test_g = distancePoint(y_test, pred_test_r)
    print(np.mean(mse_train_g), np.mean(mse_test_g))
    # plt.scatter(y_train[:, 0], y_train[:, 1])
    # plt.scatter(pred_train[:, 0], pred_train[:, 1])
    # plt.show()
    # plt.scatter(y_test[:, 0], y_test[:, 1])
    # plt.scatter(pred_test[:, 0], pred_test[:, 1])
    # plt.show()

if __name__ == "__main__":
    case1()
    # estimate()