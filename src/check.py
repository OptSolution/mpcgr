'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:20:42
LastAuthor: Please set LastEditors
lastTime: 2022-03-14 15:35:00
'''
import open3d as o3d
import os
import numpy as np
import utils
import config


def check_with_draw(pcs, match_file):
    def remove_matches(vis):
        if os.path.exists(match_file):
            os.remove(match_file)
            print("Deleted {}".format(match_file))
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="press 'D' to delete the match")
    for pc in pcs:
        vis.add_geometry(pc)
    vis.get_render_option().point_size = 1
    vis.register_key_callback(ord('D'), remove_matches)
    vis.run()
    vis.destroy_window()


def check_pairs(cfg: config.mgr_cfg):
    print("Check pairs match")
    data_name = []
    with open(os.path.join(cfg.data_path, cfg.data_file), 'r') as f:
        for line in f:
            line = line.strip("\n")
            data_name.append(line)

    for i in range(cfg.num):
        target_path = os.path.join(cfg.data_path, data_name[i])
        target_pc = o3d.io.read_point_cloud(target_path)
        target_pc.paint_uniform_color([204. / 255., 138. / 255., 77. / 255.])
        for j in range(i + 1, cfg.num):
            match_file = os.path.join(
                os.path.join(cfg.data_path, cfg.matches_folder),
                "{:03d}_{:03d}.txt".format(i, j))
            if not os.path.exists(match_file):
                continue
            matches = np.loadtxt(match_file)
            if matches.shape[0] == 0 or matches.shape[1] < 3:
                print("{} is empty".format(match_file))
                os.remove(match_file)
                continue
            tp = matches[:, 1:4]
            sp = matches[:, 5:8]
            T = utils.umeyama(sp.T, tp.T)
            source_path = os.path.join(cfg.data_path, data_name[j])
            source_pc = o3d.io.read_point_cloud(source_path)
            source_pc = source_pc.transform(T)
            source_pc.paint_uniform_color(
                [189. / 255., 189. / 255., 191. / 255.])
            print("Draw {} and {}".format(i, j))
            check_with_draw([target_pc, source_pc], match_file)

    print("\n")