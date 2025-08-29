import matplotlib.pyplot as plt
import numpy as np


def grid_pattern():
    
    plt.rcParams["figure.autolayout"] = True

    res = np.add.outer(range(15), range(21)) % 2
    plt.imshow(res, cmap="binary_r")
    # plt.xlabel('数字')
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def line_pattern():
    plt.rcParams["figure.autolayout"] = True

    res = np.zeros((16, 24))
    for i in range(res.shape[0]):
        res[i] = i%2
    print(res)
    plt.imshow(res, cmap="binary_r")
    # plt.axis('equal')
    plt.axis('off')
    plt.show()

# grid_pattern()
line_pattern()