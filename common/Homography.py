import numpy as np
from scipy.optimize import curve_fit

def svd_solve(A):
    """Solve a homogeneous least squares problem with the SVD
       method.

    Args:
       A: Matrix of constraints.
    Returns:
       The solution to the system.
    """
    U, S, V_t = np.linalg.svd(A)
    idx = np.argmin(S)

    least_squares_solution = V_t[idx]

    return least_squares_solution

def to_homogeneous(a):
    a = np.atleast_2d(a)
    a_hom = np.hstack((a, np.ones((a.shape[0],1))))

    return a_hom

def to_inhomogeneous(A):
    """Convert a stack of homogeneous vectors to an inhomogeneous
       representation.
    """
    A = np.atleast_2d(A)

    N = A.shape[0]
    A /= A[:,-1][:, np.newaxis]
    A_inhom = A[:,:-1]

    return A_inhom

def to_homogeneous_3d(A):
    """Convert a stack of inhomogeneous vectors (without a Z component)
       to a homogeneous full-form representation.
    """
    if A.ndim != 2 or A.shape[-1] != 2:
        raise ValueError('Stacked vectors must be 2D inhomogeneous')

    N = A.shape[0]
    A_3d = np.hstack((A, np.zeros((N,1))))
    A_3d_hom = to_homogeneous(A_3d)

    return A_3d_hom

def homography(pers, rect):
    assert(pers.shape[0] == rect.shape[0])
    equations = list()
    for i in range(pers.shape[0]):
        equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
        equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
        equations.append(equation1)
        equations.append(equation2)
    equations = np.array(equations)
    A = equations[:, :-1]
    b = -1 * equations[:, -1]

    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.append(h, 1)
    h = np.reshape(h, (3, 3))
    return h

def homography_svd(pers, rect):
    assert(pers.shape[0] == rect.shape[0])
    equations = list()
    for i in range(pers.shape[0]):
        equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
        equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
        equations.append(equation1)
        equations.append(equation2)

    equations = np.array(equations)
    # print(equations.shape)
    U, singular, V_transpose = np.linalg.svd(equations)
    idx = np.argmin(singular)

    h = np.reshape(V_transpose[idx], (3, 3))
    return h

def f_refine(xdata, *params):
    """Value function for Levenberg-Marquardt refinement.
    """
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params

    N = xdata.shape[0] // 2

    X = xdata[:N]
    Y = xdata[N:]

    x = (h11 * X + h12 * Y + h13) / (h31 * X + h32 * Y + h33)
    y = (h21 * X + h22 * Y + h23) / (h31 * X + h32 * Y + h33)

    result = np.zeros_like(xdata)
    result[:N] = x
    result[N:] = y

    return result


def jac_refine(xdata, *params):
    """Jacobian function for Levenberg-Marquardt refinement.
    """
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params

    N = xdata.shape[0] // 2

    X = xdata[:N]
    Y = xdata[N:]

    J = np.zeros((N * 2, 9))
    J_x = J[:N]
    J_y = J[N:]

    s_x = h11 * X + h12 * Y + h13
    s_y = h21 * X + h22 * Y + h23
    w   = h31 * X + h32 * Y + h33
    w_sq = w**2

    J_x[:, 0] = X / w
    J_x[:, 1] = Y / w
    J_x[:, 2] = 1. / w
    J_x[:, 6] = (-s_x * X) / w_sq
    J_x[:, 7] = (-s_x * Y) / w_sq
    J_x[:, 8] = -s_x / w_sq

    J_y[:, 3] = X / w
    J_y[:, 4] = Y / w
    J_y[:, 5] = 1. / w
    J_y[:, 6] = (-s_y * X) / w_sq
    J_y[:, 7] = (-s_y * Y) / w_sq
    J_y[:, 8] = -s_y / w_sq

    J[:N] = J_x
    J[N:] = J_y

    return J

def refine_homography(H, model, data):
    """Perform nonlinear least squares to refine linear homography
       estimates.

    Args:
       H: 3x3 homography matrix
       model: Nx2 world frame planar model
       data: Nx2 sensor frame correspondences
    Returns:
       Refined 3x3 homography
    """
    X, Y, x, y = model[:,0], model[:,1], data[:,0], data[:,1]

    N = X.shape[0]

    h0 = H.ravel()

    xdata = np.zeros(N * 2)
    xdata[:N] = X
    xdata[N:] = Y

    ydata = np.zeros(N * 2)
    ydata[:N] = x
    ydata[N:] = y

    # Use Levenberg-Marquardt to refine the linear homography estimate
    popt, pcov = curve_fit(f_refine, xdata, ydata, p0=h0, jac=jac_refine)
    h_refined = popt

    # Normalize and reconstitute homography
    h_refined /= h_refined[-1]
    H_refined = h_refined.reshape((3,3))

    return H_refined

def generate_v_ij(H_stack, i, j):
    """Generate intrinsic orthogonality constraints. See Zhang pg. 6 for
       details.
    """ 
    M = H_stack.shape[0]

    v_ij = np.zeros((M, 6))
    v_ij[:, 0] = H_stack[:, 0, i] * H_stack[:, 0, j]
    v_ij[:, 1] = H_stack[:, 0, i] * H_stack[:, 1, j] + H_stack[:, 1, i] * H_stack[:, 0, j]
    v_ij[:, 2] = H_stack[:, 1, i] * H_stack[:, 1, j]
    v_ij[:, 3] = H_stack[:, 2, i] * H_stack[:, 0, j] + H_stack[:, 0, i] * H_stack[:, 2, j]
    v_ij[:, 4] = H_stack[:, 2, i] * H_stack[:, 1, j] + H_stack[:, 1, i] * H_stack[:, 2, j]
    v_ij[:, 5] = H_stack[:, 2, i] * H_stack[:, 2, j]

    return v_ij

def cal_intrinsics(homographies):
    """Use computed homographies to calculate intrinsic matrix.
       Requires >= 3 homographies for a full 5-parameter intrinsic matrix.
    """
    M = len(homographies)

    # Stack homographies
    H_stack = np.zeros((M, 3, 3))
    for h, H in enumerate(homographies):
        H_stack[h] = H

    # Generate constraints
    v_00 = generate_v_ij(H_stack, 0, 0)
    v_01 = generate_v_ij(H_stack, 0, 1)
    v_11 = generate_v_ij(H_stack, 1, 1)

    # Mount constraint matrix
    V = np.zeros((2 * M, 6))
    V[:M] = v_01
    V[M:] = v_00 - v_11

    # Use SVD to solve the homogeneous system Vb = 0
    b = svd_solve(V)

    B0, B1, B2, B3, B4, B5 = b

    # Form B = K_-T K_-1
    B = np.array([[B0, B1, B3],
                  [B1, B2, B4],
                  [B3, B4, B5]])

    # Form auxilliaries
    w = B0 * B2 * B5 - B1**2 * B5 - B0 * B4**2 + 2. * B1 * B3 * B4 - B2 * B3**2
    d = B0 * B2 - B1**2

    # Use Zhang's closed form solution for intrinsic parameters (Zhang, Appendix B, pg. 18)
    v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2]) / (B[0,0] * B[1,1] - B[0,1] * B[0,1])
    lambda_ = B[2,2] - (B[0,2] * B[0,2] + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2])) / B[0,0]
    lambda_ *= -1
    print(lambda_)
    print(B[0, 0])
    alpha = np.sqrt(lambda_ / B[0,0])
    beta = alpha#np.sqrt(lambda_ * B[0,0] / (B[0,0] * B[1,1] - B[0,1] * B[0,1]))
    gamma = 0#-B[0,1] * alpha * alpha * beta / lambda_
    u0 = gamma * v0 / beta - B[0,2] * alpha * alpha / lambda_

    # Reconstitute intrinsic matrix
    K = np.array([[alpha, gamma, u0],
                  [   0.,  beta, v0],
                  [   0.,    0., 1.]])

    return K


def rectify(points, h):
    points_homo = np.column_stack([points, np.ones((points.shape[0], 1))])
    rectify = np.dot(h, points_homo.T).T
    rectify[:, 0] /= rectify[:, 2]
    rectify[:, 1] /= rectify[:, 2]
    return rectify[:, :2]

def fundamental(p1, p2):
    c1 = p2[:, 0] * p1[:, 0]
    c2 = p2[:, 0] * p1[:, 1]
    c3 = p2[:, 0]
    c4 = p1[:, 0] * p2[:, 1]
    c5 = p2[:, 1] * p1[:, 1]
    c6 = p2[:, 1]
    c7 = p1[:, 0]
    c8 = p1[:, 1]
    c9 = np.ones_like(c8)
    equations = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9])
    U, singular, V_transpose = np.linalg.svd(equations)

    F = np.reshape(V_transpose[-1], (3, 3))
    return F

def left_epipole(f):
    U, S, V = np.linalg.svd(f)
    e = V[-1]
    return e / e[2]



# ### test
# from proposed.Calibration import Calibration

import matplotlib.pyplot as plt
# from common.homography import fundamental, homography, left_epipole
def test1():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    # objp = ca.objp * 100 + 1000
    ca.homography_svd(corners, ca.objp)
    rectify = ca.rectify(corners)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(ca.objp[:, 0], ca.objp[:, 1])
    plt.scatter(rectify[:, 0], rectify[:, 1])
    plt.show()


def test2():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    objp = ca.objp
    objp[:, 0] = (objp[:, 0] - 1000)*2
    objp[:, 1] = (objp[:, 1] - 1000)
    h = ca.homography(corners, ca.objp)
    print(h)
    # print(h[2, 0] / h[2, 1])
    rectify = ca.rectify(corners)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(objp[:, 0], objp[:, 1])
    plt.scatter(rectify[:, 0], rectify[:, 1])
    plt.show()

def test3():
    import numpy as np
    objp = np.mgrid[0:11, 0:8].T.reshape((88, 2))
    objp = objp 
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    h = homography(objp, corners)
    print(h)
    objp2 = objp
    objp2[:,  0] += 1000
    objp2 = objp2*2
    # objp2[:,  0] += 1000
    h = homography(objp, objp2)
    print(h)

def test4():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    objp = ca.objp
    f_matrix = fundamental(corners, objp)
    print(f_matrix)
    e = left_epipole(f_matrix.T)
    print(e)

if __name__ == "__main__":
    test1()