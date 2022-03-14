'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:19:01
LastAuthor: Please set LastEditors
lastTime: 2022-03-14 15:32:56
'''
import numpy as np
import os
import open3d as o3d
import igraph
import config
from shutil import copyfile


def umeyama(X, Y):
    assert X.shape[0] == 3
    assert Y.shape[0] == 3
    assert X.shape[1] >= 3
    assert Y.shape[1] >= 3

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)

    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    Sxy = np.dot(Xc, Yc.T)

    U, D, VT = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = VT.T.copy()

    S = np.eye(3)
    S[2, 2] = np.linalg.det(V @ U.T)

    R = np.dot(np.dot(V, S), U.T)
    t = my - np.dot(R, mx)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


def sample(cfg: config.mgr_cfg):
    print("Sample")

    input_path = os.path.join(cfg.data_path, cfg.input_folder)
    if not os.path.exists(input_path):
        print("Not found data")
        exit()

    sample_path = os.path.join(cfg.data_path, cfg.sample_folder)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    data_name_file = open(os.path.join(cfg.data_path, cfg.data_file), 'w')
    sample_name_file = open(os.path.join(cfg.data_path, cfg.sample_file), 'w')

    data = os.listdir(input_path)
    data = sorted(data)

    for pc in data:
        print("sample {}".format(pc))
        pcd = o3d.io.read_point_cloud(os.path.join(input_path, pc))
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
        o3d.io.write_point_cloud(os.path.join(sample_path, pc), pcd)
        data_name_file.write("{}\n".format(os.path.join(cfg.input_folder, pc)))
        sample_name_file.write("{}\n".format(
            os.path.join(cfg.sample_folder, pc)))

    data_name_file.close()
    sample_name_file.close()

    print("\n")


def read_max_connected_component(cfg: config.mgr_cfg):
    if not os.path.exists(os.path.join(cfg.data_path, cfg.max_connected_file)):
        print("Cannot find max connected")
        exit()
    f = open(os.path.join(cfg.data_path, cfg.max_connected_file), 'r')
    max_cc_str = f.readline().split(" ")
    f.close()
    if "" in max_cc_str:
        max_cc_str.remove("")

    max_cc = []
    for i in max_cc_str:
        max_cc.append(int(i))

    return max_cc


def sort_for_icp(cfg: config.mgr_cfg):
    # read max connected component
    max_cc = read_max_connected_component(cfg)

    # build graph
    ug = igraph.Graph(directed=False)
    ug.add_vertices(len(max_cc))
    ug.vs['name'] = [str(i) for i in max_cc]

    # edges
    edge_list = [
        (i, j) for i in range(len(max_cc)) for j in range(i + 1, len(max_cc))
        if os.path.exists(
            os.path.join(os.path.join(cfg.data_path, cfg.matches_folder),
                         "{:03d}_{:03d}.txt".format(max_cc[i], max_cc[j])))
    ]
    ug.add_edges(edge_list)

    # sort by degree
    degs = [ug.degree(v) for v in ug.vs]
    sorted_index = sorted(range(len(max_cc)),
                          key=lambda k: degs[k],
                          reverse=True)

    # output
    sort_icp = [sorted_index[0]]
    neis = set(v['name'] for v in ug.vs[sorted_index[0]].neighbors())
    sorted_index.remove(sorted_index[0])

    while len(sorted_index) != 0:
        is_found = False
        for i in sorted_index:
            if str(max_cc[i]) in neis:
                is_found = True
                sort_icp.append(i)
                sorted_index.remove(i)
                neis = neis.union(set(v['name'] for v in ug.vs[i].neighbors()))
                break
        if not is_found:
            break

    # return
    output = [max_cc[i] for i in sort_icp]
    return output


def multi_icp(cfg: config.mgr_cfg):
    print("Multi ICP")
    result_folder = os.path.join(cfg.data_path, cfg.transform_folder)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    data_name = []
    with open(os.path.join(cfg.data_path, cfg.sample_file), 'r') as f:
        for line in f:
            line = line.strip("\n")
            data_name.append(line)

    # sort max connected component
    sort_index_icp = sort_for_icp(cfg)

    # start
    for i in sort_index_icp:
        T = copyfile(
            os.path.join(os.path.join(cfg.data_path, cfg.SDP_result_folder),
                         "{:03d}.txt".format(i)),
            os.path.join(result_folder, "{:03d}.txt".format(i)))

    if len(cfg.icp_iter) != len(cfg.icp_threshold):
        print("Iter num is not equal to threshold num!")
        exit()

    for n in range(len(cfg.icp_iter)):

        pc = o3d.io.read_point_cloud(
            os.path.join(cfg.data_path, data_name[sort_index_icp[0]]))
        T0 = np.loadtxt(
            os.path.join(result_folder,
                         "{:03d}.txt".format(sort_index_icp[0])))
        pc = pc.transform(T0)
        np.savetxt(
            os.path.join(result_folder,
                         "{:03d}.txt".format(sort_index_icp[0])), T0)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
        criteria.max_iteration = cfg.icp_iter[n]

        for i in range(1, len(sort_index_icp)):
            pci_path = os.path.join(cfg.data_path,
                                    data_name[sort_index_icp[i]])
            print("Register for {}".format(pci_path))
            pc = pc.voxel_down_sample(cfg.voxel_size)

            pci = o3d.io.read_point_cloud(pci_path)
            Ti = np.loadtxt(
                os.path.join(result_folder,
                             "{:03d}.txt".format(sort_index_icp[i])))
            pci = pci.transform(Ti)

            r = o3d.pipelines.registration.registration_icp(
                pci, pc, cfg.icp_threshold[n], np.identity(4),
                o3d.pipelines.registration.
                TransformationEstimationPointToPoint(), criteria)
            pci = pci.transform(r.transformation)
            Ti = r.transformation @ Ti
            np.savetxt(
                os.path.join(result_folder,
                             "{:03d}.txt".format(sort_index_icp[i])), Ti)

            # fusion
            points = np.vstack([np.asarray(pc.points), np.asarray(pci.points)])
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

