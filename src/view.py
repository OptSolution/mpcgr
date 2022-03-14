'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:26:04
LastAuthor: Please set LastEditors
lastTime: 2022-03-14 15:30:18
'''
import open3d as o3d
import os
import numpy as np
import config
import utils


def _view_result(cfg: config.mgr_cfg, MODE: str):
    # read max connected component
    max_cc = utils.read_max_connected_component(cfg)

    if MODE == "SDP":
        transform_path = os.path.join(cfg.data_path, cfg.SDP_result_folder)
    elif MODE == "FINAL":
        transform_path = os.path.join(cfg.data_path, cfg.transform_folder)
    elif MODE == "SPECTRAL":
        transform_path = os.path.join(cfg.data_path,
                                      cfg.spectral_result_folder)

    if not os.path.exists(transform_path):
        print("Cannot find transformation in {}".format(transform_path))
        exit()

    data_name = []
    with open(os.path.join(cfg.data_path, cfg.data_file), 'r') as f:
        for line in f:
            line = line.strip("\n")
            data_name.append(line)
    pcs = []
    for i in max_cc:
        pc_path = os.path.join(cfg.data_path, data_name[i])
        pc = o3d.io.read_point_cloud(pc_path)
        T = np.loadtxt(os.path.join(transform_path, "{:03d}.txt".format(i)))
        pc = pc.transform(T)
        pc.paint_uniform_color(np.random.rand(3))
        pcs.append(pc)

    o3d.visualization.draw_geometries(pcs)


def view_SDP(cfg: config.mgr_cfg):
    _view_result(cfg=cfg, MODE="SDP")


def view_FINAL(cfg: config.mgr_cfg):
    _view_result(cfg=cfg, MODE="FINAL")


def view_SPECTRAL(cfg: config.mgr_cfg):
    _view_result(cfg=cfg, MODE="SPECTRAL")


def view(cfg: config.mgr_cfg, folder: str):
    transform_path = os.path.join(cfg.data_path, folder)

    if not os.path.exists(transform_path):
        print("Cannot find transformation in {}".format(transform_path))
        exit()

    data_name = []
    with open(os.path.join(cfg.data_path, cfg.data_file), 'r') as f:
        for line in f:
            line = line.strip("\n")
            data_name.append(line)

    pcs = []
    for i in range(len(data_name)):
        if not os.path.exists(
                os.path.join(transform_path, "{:03d}.txt".format(i))):
            continue
        pc_path = os.path.join(cfg.data_path, data_name[i])
        pc = o3d.io.read_point_cloud(pc_path)
        T = np.loadtxt(os.path.join(transform_path, "{:03d}.txt".format(i)))
        pc = pc.transform(T)
        pcs.append(pc)

    o3d.visualization.draw_geometries(pcs)