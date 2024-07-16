import numpy as np
import cv2

def distancePoint(pa, pb):
    diff = pa - pb
    return np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

def distortion_amount(image_u, image_d):
    dist = distancePoint(image_u , image_d)
    return dist

def distortion_state(image_u, image_d):
    dist = distancePoint(image_u , image_d)
    mean = np.mean(dist)
    max = np.max(dist)
    return mean, max

def line_norm(seg):
    norm_rad = 0
    if abs(seg[-1, 1] - seg[0, 1]) <= abs(seg[-1, 0] - seg[0, 0]):
        params = np.polyfit(seg[:, 0], seg[:, 1], 1)
        norm_rad = np.arctan(params[0]) + np.pi / 2
        if norm_rad > np.pi / 2:
            norm_rad -= np.pi
    else:
        params = np.polyfit(seg[:, 1], seg[:, 0], 1)
        norm_rad = -np.arctan(params[0])
    return norm_rad

def neibour_line_fit(points:np.ndarray, center):
    """
        returns the normal vector angle of a line
    """
    NEIBOUR_COUNT = 100
    diff = np.array(points)
    diff[:, 0] -= center[0]
    diff[:, 1] -= center[1]
    dist = diff[:, 0]**2 + diff[:, 1]**2
    minidx = np.argmin(dist)
    seg = points[minidx - NEIBOUR_COUNT:minidx + NEIBOUR_COUNT+1]
    norm_rad = line_norm(seg)
    a = np.cos(norm_rad)
    b = np.sin(norm_rad)
    return norm_rad




def twoPointCosSin(p1, p2):
    d = np.linalg.norm(p1 - p2)
    costheta = (p2[0] - p1[0]) / d
    sintheta = (p2[1] - p1[1]) / d
    return costheta, sintheta

def perspective():
    import cv2
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[50, 20], [500, 20],
                       [50, 640], [500, 640]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    print(matrix)

def rod_mat2angles():
    rad1 = 0#np.pi / 6  #z
    rad2 = np.pi / 6  #y
    rad3 = np.pi / 6  #x
    mat1 = [np.cos(rad1), -1 * np.sin(rad1), 0, np.sin(rad1), np.cos(rad1), 0, 0, 0, 1]
    mat2 = [np.cos(rad2), 0, -1 * np.sin(rad2), 0, 1, 0, np.sin(rad2), 0, np.cos(rad2)]
    mat3 = [1, 0, 0, 0, np.cos(rad3), -1 * np.sin(rad3), 0, np.sin(rad3), np.cos(rad3)]
    mat1 = np.reshape(np.array(mat1), (3, 3))
    mat2 = np.reshape(np.array(mat2), (3, 3))
    mat3 = np.reshape(np.array(mat3), (3, 3))
    mat = np.dot(mat2, mat3)
    mat = np.dot(mat1, mat)
    rvecs = cv2.Rodrigues(mat)
    print(rvecs[0])

def rod_angles2mat():
    rad1 = 0#np.pi / 6  #z
    rad2 = np.pi / 60  #y
    rad3 = 0#np.pi / 60  #z
    rvecs = np.array([rad1, rad2, rad3])
    rvecs = np.reshape(rvecs, (3, 1))
    
    mat = cv2.Rodrigues(rvecs)
    print(mat[0])
# rod_mat2angles()
# rod_angles2mat()

if __name__ == "__main__":
    # rod_mat2angles()
    rod_angles2mat()
    # perspective()