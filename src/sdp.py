'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:23:34
LastAuthor: Please set LastEditors
lastTime: 2022-02-22 13:51:16
'''
import cvxpy as cp
import numpy as np
import os
import config
import utils


def SDP(C: np.ndarray, nums: int, SDP_path: str):
    print("SDP")
    G = cp.Variable((3 * nums, 3 * nums), symmetric=True)

    constraints = [G >> 0]
    for i in range(nums):
        z1 = np.zeros((3 * nums, 3 * nums))
        z1[3 * i, 3 * i] = 1.
        z2 = np.zeros((3 * nums, 3 * nums))
        z2[3 * i + 1, 3 * i + 1] = 1.
        z3 = np.zeros((3 * nums, 3 * nums))
        z3[3 * i + 2, 3 * i + 2] = 1.
        constraints += [
            cp.trace(z1 @ G) == 1.,
            cp.trace(z2 @ G) == 1.,
            cp.trace(z3 @ G) == 1.
        ]

        z4 = np.zeros((3 * nums, 3 * nums))
        z4[3 * i + 1, 3 * i] = 1.
        z4[3 * i, 3 * i + 1] = 1.
        z5 = np.zeros((3 * nums, 3 * nums))
        z5[3 * i + 1, 3 * i + 2] = 1.
        z5[3 * i + 2, 3 * i + 1] = 1.
        z6 = np.zeros((3 * nums, 3 * nums))
        z6[3 * i, 3 * i + 2] = 1.
        z6[3 * i + 2, 3 * i] = 1.
        constraints += [
            cp.trace(z4 @ G) == 0.,
            cp.trace(z5 @ G) == 0.,
            cp.trace(z6 @ G) == 0.
        ]

    prob = cp.Problem(cp.Minimize(cp.trace(C @ G)), constraints)
    prob.solve(verbose=True)

    print("status:", prob.status)
    print("optimal value", prob.value)

    np.savetxt(os.path.join(SDP_path, "G.txt"), G.value)
    print("\n")

    return G.value


def rounding(G: np.ndarray, max_cc: list, SDP_path: str, transform_path: str,
             nums: int, B: np.ndarray, L_inv: np.ndarray):
    print("Rounding")
    if not os.path.exists(transform_path):
        os.mkdir(transform_path)

    # compute W
    e_val, e_vec = np.linalg.eig(G)
    sorted_indices = np.argsort(e_val)

    w_val = e_val[sorted_indices[-3:]]
    w_vec = e_vec[:, sorted_indices[-3:]]
    w_val = np.sqrt(w_val)

    W = np.stack((w_val[-1] * w_vec[:, -1], w_val[-2] * w_vec[:, -2],
                  w_val[-3] * w_vec[:, -3]))

    # SVD for O
    O = np.zeros((3, 3 * nums))
    for i in range(nums):
        Wi = W[:, 3 * i:3 * i + 3]
        ui, si, vhi = np.linalg.svd(Wi)
        O[:, 3 * i:3 * i + 3] = ui @ vhi
    np.savetxt(os.path.join(SDP_path, "O.txt"), O)

    # Z
    Z = -O @ B @ L_inv
    np.savetxt(os.path.join(SDP_path, "Z.txt"), Z)

    # result
    T0 = np.identity(4)
    T0[0:3, 0:3] = O[:, 0:3]
    T0[0:3, 3] = Z[:, 0]
    T0_inv = np.linalg.inv(T0)
    for i in range(nums):
        Ti = np.identity(4)
        Ti[0:3, 0:3] = O[:, 3 * i:3 * i + 3]
        Ti[0:3, 3] = Z[:, i]
        Ti = T0_inv @ Ti
        np.savetxt(
            os.path.join(transform_path, "{:03d}.txt".format(max_cc[i])), Ti)

    print("\n")


def mc_SDP(cfg: config.mgr_cfg):
    cfg.update_max_connected()

    print("BUild SDP")
    SDP_path = os.path.join(cfg.data_path, cfg.SDP_folder)
    if not os.path.exists(SDP_path):
        os.mkdir(SDP_path)

    # read max connected component
    max_cc = utils.read_max_connected_component(cfg)

    # set matrix
    L = np.zeros((cfg.mc_num, cfg.mc_num))
    B = np.zeros((3 * cfg.mc_num, cfg.mc_num))
    D = np.zeros((3 * cfg.mc_num, 3 * cfg.mc_num))
    L_inv = np.zeros((cfg.mc_num, cfg.mc_num))
    C = np.zeros((3 * cfg.mc_num, 3 * cfg.mc_num))

    # update matrix
    for i in range(cfg.mc_num):
        for j in range(i + 1, cfg.mc_num):
            file_path_ = os.path.join(
                os.path.join(cfg.data_path, cfg.matches_folder),
                "{:03d}_{:03d}.txt".format(max_cc[i], max_cc[j]))
            if os.path.exists(file_path_):
                e_ij = np.zeros((cfg.mc_num, 1))
                e_ij[i, 0] = 1.0
                e_ij[j, 0] = -1.0

                ei_m = np.zeros((3 * cfg.mc_num, 3))
                ej_m = np.zeros((3 * cfg.mc_num, 3))
                ei_m[3 * i:3 * i + 3, :] = np.identity(3)
                ej_m[3 * j:3 * j + 3, :] = np.identity(3)

                matches_ = np.loadtxt(file_path_)
                pis = matches_[:, 1:4]
                pjs = matches_[:, 5:8]

                for k in range(pis.shape[0]):
                    d = ei_m @ pis[k, :] - ej_m @ pjs[k, :]
                    d = np.reshape(d, (3 * cfg.mc_num, 1))
                    dD = d @ d.T
                    dB = d @ e_ij.T
                    dL = e_ij @ e_ij.T

                    L += dL
                    B += dB
                    D += dD

    # cal L inverse
    num_inv = 1 / float(cfg.mc_num)
    e = np.ones((cfg.mc_num, 1))
    eet = e @ e.T * num_inv

    L_inv = L + eet
    L_inv = np.linalg.inv(L_inv)
    L_inv = L_inv - eet

    C = D - B @ L_inv @ B.T

    np.savetxt(os.path.join(SDP_path, "L.txt"), L)
    np.savetxt(os.path.join(SDP_path, "B.txt"), B)
    np.savetxt(os.path.join(SDP_path, "D.txt"), D)
    np.savetxt(os.path.join(SDP_path, "L_inv.txt"), L_inv)
    np.savetxt(os.path.join(SDP_path, "C.txt"), C)

    G = SDP(C, cfg.mc_num, SDP_path)
    rounding(G, max_cc, SDP_path,
             os.path.join(cfg.data_path, cfg.SDP_result_folder), cfg.mc_num, B,
             L_inv)


def mc_spectral(cfg: config.mgr_cfg):
    cfg.update_max_connected()

    print("Spectral")
    Spectral_path = os.path.join(cfg.data_path, cfg.spectral_folder)
    if not os.path.exists(Spectral_path):
        os.mkdir(Spectral_path)

    # read max connected component
    max_cc = utils.read_max_connected_component(cfg)

    # set matrix
    L = np.zeros((cfg.mc_num, cfg.mc_num))
    B = np.zeros((3 * cfg.mc_num, cfg.mc_num))
    D = np.zeros((3 * cfg.mc_num, 3 * cfg.mc_num))
    L_inv = np.zeros((cfg.mc_num, cfg.mc_num))
    C = np.zeros((3 * cfg.mc_num, 3 * cfg.mc_num))

    # update matrix
    for i in range(cfg.mc_num):
        for j in range(i + 1, cfg.mc_num):
            file_path_ = os.path.join(
                os.path.join(cfg.data_path, cfg.matches_folder),
                "{:03d}_{:03d}.txt".format(max_cc[i], max_cc[j]))
            if os.path.exists(file_path_):
                e_ij = np.zeros((cfg.mc_num, 1))
                e_ij[i, 0] = 1.0
                e_ij[j, 0] = -1.0

                ei_m = np.zeros((3 * cfg.mc_num, 3))
                ej_m = np.zeros((3 * cfg.mc_num, 3))
                ei_m[3 * i:3 * i + 3, :] = np.identity(3)
                ej_m[3 * j:3 * j + 3, :] = np.identity(3)

                matches_ = np.loadtxt(file_path_)
                pis = matches_[:, 1:4]
                pjs = matches_[:, 5:8]

                for k in range(pis.shape[0]):
                    d = ei_m @ pis[k, :] - ej_m @ pjs[k, :]
                    d = np.reshape(d, (3 * cfg.mc_num, 1))
                    dD = d @ d.T
                    dB = d @ e_ij.T
                    dL = e_ij @ e_ij.T

                    L += dL
                    B += dB
                    D += dD

    # cal L inverse
    num_inv = 1 / float(cfg.mc_num)
    e = np.ones((cfg.mc_num, 1))
    eet = e @ e.T * num_inv

    L_inv = L + eet
    L_inv = np.linalg.inv(L_inv)
    L_inv = L_inv - eet

    C = D - B @ L_inv @ B.T

    np.savetxt(os.path.join(Spectral_path, "L.txt"), L)
    np.savetxt(os.path.join(Spectral_path, "B.txt"), B)
    np.savetxt(os.path.join(Spectral_path, "D.txt"), D)
    np.savetxt(os.path.join(Spectral_path, "L_inv.txt"), L_inv)
    np.savetxt(os.path.join(Spectral_path, "C.txt"), C)

    # W
    e_val, e_vec = np.linalg.eig(C)
    sorted_indices = np.argsort(e_val)

    W = np.stack((e_vec[:, sorted_indices[0]], e_vec[:, sorted_indices[1]],
                  e_vec[:, sorted_indices[2]]))
    W *= np.sqrt(float(cfg.mc_num))

    # O
    O = np.zeros((3, 3 * cfg.mc_num))
    for i in range(cfg.mc_num):
        Wi = W[:, 3 * i:3 * i + 3]
        ui, si, vhi = np.linalg.svd(Wi)
        O[:, 3 * i:3 * i + 3] = ui @ vhi
    np.savetxt(os.path.join(Spectral_path, "O.txt"), O)

    # Z
    Z = -O @ B @ L_inv
    np.savetxt(os.path.join(Spectral_path, "Z.txt"), Z)

    # result
    transform_path = os.path.join(cfg.data_path, cfg.spectral_result_folder)
    if not os.path.exists(transform_path):
        os.mkdir(transform_path)

    T0 = np.identity(4)
    T0[0:3, 0:3] = O[:, 0:3]
    T0[0:3, 3] = Z[:, 0]
    T0_inv = np.linalg.inv(T0)
    for i in range(cfg.mc_num):
        Ti = np.identity(4)
        Ti[0:3, 0:3] = O[:, 3 * i:3 * i + 3]
        Ti[0:3, 3] = Z[:, i]
        Ti = T0_inv @ Ti
        np.savetxt(
            os.path.join(transform_path, "{:03d}.txt".format(max_cc[i])), Ti)

    print("\n")
