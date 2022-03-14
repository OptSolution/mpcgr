'''
Description: 
Author: Chen Wang
Email: mr_cwang@foxmail.com
since: 2021-12-13 14:44:39
LastAuthor: Please set LastEditors
lastTime: 2022-02-22 13:45:12
'''
import os
import igraph


class mgr_cfg:
    def __init__(self, data_path: str) -> None:
        self.input_folder = "data"
        self.sample_folder = "sample"
        self.data_file = "data.txt"
        self.sample_file = "sample.txt"
        self.matches_folder = "matches"
        self.SDP_folder = "SDP_matrix"
        self.SDP_result_folder = "SDP_result"
        self.spectral_folder = "spectral_matrix"
        self.spectral_result_folder = "spectral_result"
        self.transform_folder = "result"
        self.max_connected_file = "max_connect_components.txt"
        self.data_path = data_path

        self.num = self._pc_num()

        self.voxel_size = 0.05
        self.noise_bound = 0.05
        self.icp_threshold = [0.05]
        self.icp_iter = [50]

        self.mc_num = -1
        self.is_mc = False

        # for pairs match
        self.is_visual = False
        self.is_write_match = True
        self.is_visual_pair_result = False
        self.graph_max_vertex_num = 200
        self.min_inliers_num = 10
        self.fitness_dist = 0.05
        self.min_fitness = 0.3

    def _pc_num(self):
        input_path = os.path.join(self.data_path, self.input_folder)
        if not os.path.exists(input_path):
            print("Not found data")
            exit()

        data = os.listdir(input_path)
        return len(data)

    def set_voxel_size(self, voxel_size: float):
        self.voxel_size = voxel_size

    def set_noise_bound(self, noise_bound: float):
        self.noise_bound = noise_bound

    def set_icp_threshold(self, icp_threshold: list):
        self.icp_threshold = icp_threshold

    def set_icp_iter(self, icp_iter: list):
        self.icp_iter = icp_iter

    def _find_max_connected(self):
        if not os.path.exists(os.path.join(self.data_path,
                                           self.matches_folder)):
            print("Cannot find matches")
            exit()

        g = igraph.Graph(directed=False)
        for i in range(self.num):
            g.add_vertex(str(i))

        edge_list = []
        for i in range(self.num):
            for j in range(i + 1, self.num):
                if os.path.exists(
                        os.path.join(
                            os.path.join(self.data_path, self.matches_folder),
                            "{:03d}_{:03d}.txt".format(i, j))):
                    edge_list.append((str(i), str(j)))
        g.add_edges(edge_list)

        cc = g.decompose(minelements=2)

        max_count = 0
        max_cc = []
        for c in cc:
            if c.vcount() > max_count:
                max_count = c.vcount()
                max_cc.clear()
                for v in c.vs:
                    max_cc.append(v['name'])
        print("The max connected component is {}".format(max_cc))

        f = open(os.path.join(self.data_path, self.max_connected_file), 'w')
        for i in max_cc:
            f.write(i)
            f.write(" ")
        f.close()

        return max_count

    def update_max_connected(self):
        if self.is_mc:
            return

        self.mc_num = self._find_max_connected()
        self.is_mc = True

    def set_visual(self, is_visual: bool):
        self.is_visual = is_visual

    def set_max_vertex(self, vertex_num: int):
        self.graph_max_vertex_num = vertex_num

    def set_min_inlier_num(self, inlier_num: int):
        self.min_inliers_num = inlier_num

    def set_fitness_dist(self, dist: float):
        self.fitness_dist = dist

    def set_min_fitness(self, min_fitness: float):
        self.min_fitness = min_fitness

    def set_write_pair_match(self, is_write: bool):
        self.is_write_match = is_write

    def set_visual_pair_result(self, is_visual: bool):
        self.is_visual_pair_result = is_visual
