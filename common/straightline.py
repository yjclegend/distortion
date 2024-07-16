import pickle

import numpy as np

from common.chessboard import findchessboard

def straightness(line, normalize=False):
    """
    calculate the mean residual of a line fit to reflect straightness of undistorted lines
    must be called after undistort

    """
    lx = line[:, 0]
    ly = line[:, 1]
    if abs(lx[0] - lx[-1]) >= abs(ly[0] - ly[-1]):
        x, y = lx, ly
    else:
        x, y = ly, lx
    params = np.polyfit(x, y, 1)

    pred = params[0] * x + params[1]
    rmse = np.sqrt(np.mean((pred-y)**2))
    if normalize:
        dist = np.sqrt((line[0, 0] - line[-1, 0])**2 + (line[0, 1] - line[-1, 1])**2)
        rmse /= dist
    # return np.mean((pred-y)**2)
    return rmse

def load_lines(name, smooth=True, flat=True, zoning=None):
    '''
        smooth: use quadraplicte smoothing
        
    '''
    if smooth:
        f = open('data/' + "smooth_" + name[5:] + '.pkl', 'rb')
    else:
        f = open('data/' + "lines_" + name[5:] + '.pkl', 'rb')
    line_groups = pickle.load(f)
    lines = list()
    if flat:
        for line_set in line_groups:
            if zoning is None:
                lines.extend(line_set)
            else:
                left, right, bot, top = zoning
                for line in line_set:
                    # import matplotlib.pyplot as plt
                    # plt.scatter(line[:, 0], line[:, 1])
                    line = line[np.where((line[:, 0] > left) & (line[:, 0] < right) & (line[:, 1] > bot) & (line[:, 1] < top))]
                    # plt.scatter(line[:, 0], line[:, 1])
                    # plt.show()
                    # exit()
                    if len(line) > 3000:
                        lines.append(line)
        return lines
    else:
        return line_groups

def line_from_grid(file_path, size):
    corners = findchessboard(filepath=file_path, size=size)
    row, col = size

    # horizontal lines
    hori_lines = list()
    for r in range(row):
        line = corners[r:None:row]
        hori_lines.append(line)
    
    # vertical lines
    vert_lines = list()
    for c in range(col):
        line = corners[c*row:(c+1) * row]
        vert_lines.append(line)
    return hori_lines , vert_lines