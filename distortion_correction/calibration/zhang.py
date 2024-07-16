import numpy as np

import common.homography as homography
from common.chessboard import create_ground_truth

import extrinsics, distortion, refinement

def zhang_calibration(model, all_data):
    homographies = []
    # all_data.pop(2)
    for data in all_data:
        H = homography.homography_svd(data, model)
        H = homography.refine_homography(H, model, data)
        # print(H)
        # H = homography.calculate_homography(model, data)
        # H = homography.refine_homography(H, model, data)
        # print(H)
        # exit()
        homographies.append(H)
    K = homography.cal_intrinsics(homographies)
    print(K)

    model_hom_3d = homography.to_homogeneous_3d(model)
    # Compute extrinsics based on fixed intrinsics
    extrinsic_matrices = []
    for h, H in enumerate(homographies):
        E = extrinsics.recover_extrinsics(H, K)
        extrinsic_matrices.append(E)

        # Form projection matrix
        # P = np.dot(K, E)

        # predicted = np.dot(model_hom_3d, P.T)
        # predicted = to_inhomogeneous(predicted)
        # data = all_data[h]
        # nonlinear_sse_decomp = np.sum((predicted - data)**2)
    
    # Calculate radial distortion based on fixed intrinsics and extrinsics
    k = distortion.calculate_lens_distortion(model, all_data, K, extrinsic_matrices)

    # Nonlinearly refine all parameters(intrinsics, extrinsics, and distortion)
    K_opt, k_opt, extrinsics_opt = refinement.refine_all_parameters(model, all_data, K, k, extrinsic_matrices)

    return K_opt, k_opt, extrinsics_opt


if __name__ == "__main__":
    model = create_ground_truth((11, 8))
    import pickle
    f = open('data/chessboard20240416.pkl', 'rb')
    all_data = pickle.load(f)
    K_opt, k_opt, extrinsics_opt = zhang_calibration(model, all_data)
    print("intrinsics: " , K_opt)
    print('dist: ', k_opt)
    # print('extrinsics:')