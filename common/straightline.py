import pickle

import numpy as np

from common.chessboard import findchessboard

def straightness(line, normalize=True):
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
    """
    Extract horizontal and vertical lines from a grid of chessboard corners and convert each line to a NumPy ndarray.

    Args:
        file_path (str): The path to the file containing chessboard corner data.
        size (tuple): A tuple (num_rows, num_cols) representing the number of rows and columns in the grid.

    Returns:
        tuple: A tuple containing two lists:
            - horizontal_lines (list of ndarrays): Each ndarray contains the corners of one horizontal line.
            - vertical_lines (list of ndarrays): Each ndarray contains the corners of one vertical line.
    
    Example:
        >>> file_path = 'path/to/chessboard_corners.txt'
        >>> size = (6, 9)  # Example grid size with 6 rows and 9 columns
        >>> horizontal_lines, vertical_lines = line_from_grid(file_path, size)
        >>> print(horizontal_lines)
        [array([corner1, corner2, corner3, ...]), array([corner11, corner12, corner13, ...]), ...]
        >>> print(vertical_lines)
        [array([corner1, corner11, corner21, ...]), array([corner2, corner12, corner22, ...]), ...]
    """
    # Find chessboard corners using the provided file path and size
    corners = findchessboard(filepath=file_path, size=size)
    assert(corners is not None)
    # Unpack the grid size into number of rows and columns
    num_rows, num_cols = size

    # Extract horizontal lines and convert to ndarrays
    horizontal_lines = [
        np.array(corners[row_index:len(corners):num_rows])
        for row_index in range(num_rows)
    ]

    # Extract vertical lines and convert to ndarrays
    vertical_lines = [
        np.array(corners[col_index * num_rows:(col_index + 1) * num_rows])
        for col_index in range(num_cols)
    ]

    return horizontal_lines, vertical_lines