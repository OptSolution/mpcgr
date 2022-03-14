'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-11-08 16:38:08
LastAuthor: Please set LastEditors
lastTime: 2022-03-14 16:16:37
'''
import os
import sys
import json

pwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(pwd, "src"))
import match
import utils
import check
import sdp
import view
import config


def main(cfg: config.mgr_cfg):
    utils.sample(cfg)
    match.pair_match(cfg)
    check.check_pairs(cfg)
    sdp.mc_SDP(cfg)
    # view.view_SDP(cfg)
    utils.multi_icp(cfg)
    view.view_FINAL(cfg)


if __name__ == "__main__":
    data_path = os.path.join(pwd, 'test_data')
    config_path = os.path.join(data_path, "config.json")

    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    if len(sys.argv) >= 3:
        config_path = sys.argv[2]

    cfg = config.mgr_cfg(data_path)
    # check config file
    if os.path.exists(config_path):
        print("Read config")
        config_file = {}
        with open(config_path, 'r') as f:
            config_file = json.load(f)
        if "voxel_size" in config_file:
            cfg.set_voxel_size(config_file["voxel_size"])
        if "noise_bound" in config_file:
            cfg.set_noise_bound(config_file["noise_bound"])
        if "icp_threshold" in config_file:
            cfg.set_icp_threshold(config_file["icp_threshold"])
        if "icp_iter" in config_file:
            cfg.set_icp_iter(config_file["icp_iter"])
        if "graph_max_vertex_num" in config_file:
            cfg.set_max_vertex(config_file["graph_max_vertex_num"])
        if "min_inliers_num" in config_file:
            cfg.set_min_inlier_num(config_file["min_inliers_num"])
        if "fitness_dist" in config_file:
            cfg.set_fitness_dist(config_file["fitness_dist"])
        if "min_fitness" in config_file:
            cfg.set_min_fitness(config_file["min_fitness"])
    main(cfg)
