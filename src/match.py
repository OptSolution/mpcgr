'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:13:32
LastAuthor: Please set LastEditors
lastTime: 2022-01-05 16:50:11
'''

import os
import config
import open3d as o3d
import igraph
import numpy as np
import utils
import signal
import time


class reg_result:
    def __init__(self) -> None:
        self.is_success = False
        self.transform = np.identity(4)
        self.fitness = 0.0
        self.target_inliers = []
        self.source_inliers = []
        self.target_p = []
        self.source_p = []


def write_result(result: reg_result, path: str):
    match_write = open(path, 'w')
    for c in range(len(result.source_inliers)):
        match_write.write("{} {} {} {} {} {} {} {}\n".format(
            result.target_inliers[c], result.target_p[c, 0],
            result.target_p[c, 1], result.target_p[c, 2],
            result.source_inliers[c], result.source_p[c, 0],
            result.source_p[c, 1], result.source_p[c, 2]))
    match_write.close()


def pair_match(cfg: config.mgr_cfg):

    match_path = os.path.join(cfg.data_path, cfg.matches_folder)
    if os.path.exists(match_path):
        match_list = os.listdir(match_path)
        for match_file in match_list:
            os.remove(os.path.join(match_path, match_file))
        os.removedirs(match_path)
    os.mkdir(match_path)

    data_name = []
    with open(os.path.join(cfg.data_path, cfg.sample_file), 'r') as f:
        for line in f:
            line = line.strip("\n")
            data_name.append(line)

    for i in range(cfg.num):
        target_path = os.path.join(cfg.data_path, data_name[i])
        target = o3d.io.read_point_cloud(target_path)
        for j in range(i + 1, cfg.num):
            source_path = os.path.join(cfg.data_path, data_name[j])
            source = o3d.io.read_point_cloud(source_path)

            print("Match {} and {}".format(data_name[i], data_name[j]))
            rr = graph_match(cfg, target, source)

            if rr.is_success and cfg.is_write_match:
                write_result(
                    rr,
                    os.path.join(match_path, "{:03d}_{:03d}.txt".format(i, j)))

            print("\n")

    print("\n")


def mutual_match(target_feature, source_feature):
    # source - target
    target_kdtree = o3d.geometry.KDTreeFlann(target_feature)
    target_corres = []
    for i in range(source_feature.num()):
        [_, idx,
         _] = target_kdtree.search_knn_vector_xd(source_feature.data[:, i], 1)
        target_corres.append((i, idx[0]))

    source_kdtree = o3d.geometry.KDTreeFlann(source_feature)
    source_corres = []
    for i in range(target_feature.num()):
        [_, idx,
         _] = source_kdtree.search_knn_vector_xd(target_feature.data[:, i], 1)
        source_corres.append((idx[0], i))

    # mutual
    corres_mutual = [
        c for c in target_corres if source_corres[c[1]][0] == c[0]
    ]

    return corres_mutual


def max_cliques(graph):
    mc_ = graph.largest_cliques()
    mc = [[int(graph.vs['name'][mc_[i][j]]) for j in range(len(mc_[i]))]
          for i in range(len(mc_))]
    return mc


def build_graph(corres: list, source_pc, target_pc, max_dist: float,
                max_vertex: int):
    ug = igraph.Graph(directed=False)
    ug.add_vertices(len(corres))
    ug.vs['name'] = [str(i) for i in range(len(corres))]

    # add edges
    edge_list_pair = [(corres[i][0], corres[j][0], corres[i][1], corres[j][1])
                      for i in range(len(corres))
                      for j in range(i + 1, len(corres))]
    edge_list_np = np.array(edge_list_pair)
    sp1s = np.asarray(source_pc.points)[edge_list_np[:, 0], :]
    sp2s = np.asarray(source_pc.points)[edge_list_np[:, 1], :]
    diff_s = np.linalg.norm(sp1s - sp2s, axis=1)

    tp1s = np.asarray(target_pc.points)[edge_list_np[:, 2], :]
    tp2s = np.asarray(target_pc.points)[edge_list_np[:, 3], :]
    diff_t = np.linalg.norm(tp1s - tp2s, axis=1)

    diff_abs = np.abs(diff_t - diff_s)
    index = np.where(diff_abs < max_dist)

    edge_list_full = [(i, j) for i in range(len(corres))
                      for j in range(i + 1, len(corres))]
    edge_list = [edge_list_full[i] for i in index[0]]
    ug.add_edges(edge_list)

    print("Origin graph size is {} vertices and {} edges.".format(
        str(ug.vcount()), str(ug.ecount())))

    # filter
    if ug.vcount() > max_vertex:
        degs = [ug.degree(v) for v in ug.vs]
        sorted_index = sorted(range(len(degs)),
                              key=lambda k: degs[k],
                              reverse=True)
        g_sub = ug.subgraph(sorted_index[:max_vertex])
        return g_sub

    return ug


def filter_inliers(fusion_matches, source, target, max_dist):
    inliers = []
    fusion_matches_np = np.array(fusion_matches)
    for i in range(len(fusion_matches)):
        for j in range(i + 1, len(fusion_matches)):
            for k in range(j + 1, len(fusion_matches)):
                sps = np.asarray(source.points)[[
                    fusion_matches[i][0], fusion_matches[j][0],
                    fusion_matches[k][0]
                ], :]
                tps = np.asarray(target.points)[[
                    fusion_matches[i][1], fusion_matches[j][1],
                    fusion_matches[k][1]
                ], :]
                T = utils.umeyama(sps.T, tps.T)

                spsm = np.asarray(source.points)[fusion_matches_np[:, 0], :]
                tpsm = np.asarray(target.points)[fusion_matches_np[:, 1], :]
                dists = np.linalg.norm(T[0:3, 0:3] @ spsm.T +
                                       T[0:3, 3].reshape(3, 1) - tpsm.T,
                                       axis=0)
                temp_inliers = [
                    fusion_matches[i] for i in range(len(fusion_matches))
                    if dists[i] < max_dist
                ]

                if len(temp_inliers) > len(inliers):
                    inliers.clear()
                    inliers = temp_inliers.copy()

    return inliers


def time_limit(set_time, callback):
    def wraps(func):
        def handler(signum, frame):
            raise RuntimeError()

        def deco(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(set_time)
                res = func(*args, **kwargs)
                signal.alarm(0)
                return res

            except RuntimeError as e:
                callback()

        return deco

    return wraps


def after_timeout():
    print("Time out!")
    return


@time_limit(60 * 60, after_timeout)
def graph_match(cfg: config.mgr_cfg, target, source) -> reg_result:
    # source and target both have normals

    if cfg.is_visual:
        target.paint_uniform_color([204. / 255., 138. / 255., 77. / 255.])
        source.paint_uniform_color([189. / 255., 189. / 255., 191. / 255.])
        o3d.visualization.draw_geometries([source, target],
                                          width=640,
                                          height=480)

    # compute feature
    target_feature = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamKNN())
    source_feature = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamKNN())

    # match
    corres_mutual = mutual_match(target_feature, source_feature)
    print("Match size is {}".format(len(corres_mutual)))

    if cfg.is_visual:
        origin_corrs_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            source, target, corres_mutual)
        origin_corrs_lines.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries([source, target, origin_corrs_lines],
                                          width=640,
                                          height=480)

    # build graph
    g = build_graph(corres_mutual, source, target, cfg.noise_bound,
                    cfg.graph_max_vertex_num)
    print("Graph size is {} vertices and {} edges.".format(
        str(g.vcount()), str(g.ecount())))

    # connected components
    st = time.time()
    # max clique
    mc = max_cliques(g)
    print("Max clique num is {}, size is {}".format(len(mc), len(mc[0])))
    if len(mc[0]) < cfg.min_inliers_num:
        return reg_result()

    # fusion
    fusion_matches_index = {i for c in mc for i in c}
    fusion_matches = [corres_mutual[i] for i in fusion_matches_index]
    print("Fusion size is {}".format(len(fusion_matches_index)))
    et = time.time()
    print("Max cliques cost {} s".format(et - st))

    # filter
    inliers_matches = filter_inliers(fusion_matches, source, target,
                                     cfg.noise_bound)
    print("Inliers size is {}".format(len(inliers_matches)))
    if len(inliers_matches) < cfg.min_inliers_num:
        return reg_result()

    if cfg.is_visual:
        inliers_corrs_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            source, target, inliers_matches)
        inliers_corrs_lines.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries(
            [source, target, inliers_corrs_lines], width=640, height=480)

    # inliers
    source_inliers = [i[0] for i in inliers_matches]
    target_inliers = [i[1] for i in inliers_matches]
    source_p = np.asarray(source.points)[source_inliers, :]
    target_p = np.asarray(target.points)[target_inliers, :]

    # global transform
    T = utils.umeyama(source_p.T, target_p.T)
    print("Global transformation is:")
    print(T)
    source = source.transform(T)

    if cfg.is_visual or cfg.is_visual_pair_result:
        o3d.visualization.draw_geometries([source, target],
                                          width=640,
                                          height=480)

    # fitness
    r = o3d.pipelines.registration.evaluate_registration(
        source, target, cfg.fitness_dist, np.identity(4))
    print("Fitness is {}".format(r.fitness))
    if r.fitness < cfg.min_fitness:
        return reg_result()

    rr = reg_result()
    rr.is_success = True
    rr.source_inliers = source_inliers
    rr.target_inliers = target_inliers
    rr.source_p = source_p
    rr.target_p = target_p
    rr.fitness = r.fitness
    rr.transform = r.transformation @ T
    return rr
