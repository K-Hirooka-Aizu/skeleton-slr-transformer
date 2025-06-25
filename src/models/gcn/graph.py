import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'mediapipe_hand':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(0,1),(1,2),(2,3),(3,4),
                             (0,5),(5,6),(6,7),(7,8),
                             (0,9),(9,10),(10,11),(11,12),
                             (0,13),(13,14),(14,15),(15,16),
                             (0,17),(17,18),(18,19),(19,20)]
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
            
        elif layout == 'wlasl_openpose':
            self.num_node = 55
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(0,1),
                              (1,2),(2,3),(3,4),
                              (1,5),(5,6),(6,7),
                              (1,8),
                              (0,9),(0,10),(9,11),(10,12),
                              # left hand 13~33
                              (7,13),
                            #   (4,13),
                              (13,14), (14,15), (15,16), (16,17),
                              (13,18), (18,19), (19,20), (20,21),
                              (13,22), (22,23), (23,24), (24,25),
                              (13,26), (26,27), (27,28), (28,29),
                              (13,30), (30,31), (31,32), (32,33),
                              # right hand 34~54
                              (4,34),
                            #   (7,34),
                              (34,35), (35,36), (36,37), (37,38),
                              (34,39), (39,40), (40,41), (41,42),
                              (34,43), (43,44), (44,45), (45,46),
                              (34,47), (47,48), (48,49), (49,50),
                              (34,51), (51,52), (52,53), (53,54),
                             ]
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1

        elif layout == 'mmpose_coco_whole':
            self.num_node = 55
            self_link = [(i, i) for i in range(self.num_node)]
            rh_idx = 14
            lh_idx = 35
            neighbor_1base = [
                (1,2),(2,4),
                (1,3),(3,5),
                (6,8),(8,10),
                (7,9),(9,11),
                (1,12),(12,6),(12,7),(12,13),
                
                # left hand 12~32
                
                (10,rh_idx),
                (rh_idx,rh_idx+1), (rh_idx+1,rh_idx+2), (rh_idx+2,rh_idx+3), (rh_idx+3,rh_idx+4),
                (rh_idx,rh_idx+5), (rh_idx+5,rh_idx+6), (rh_idx+6,rh_idx+7), (rh_idx+7,rh_idx+8),
                (rh_idx,rh_idx+9), (rh_idx+9,rh_idx+10), (rh_idx+10,rh_idx+11), (rh_idx+11,rh_idx+12),
                (rh_idx,rh_idx+13), (rh_idx+13,rh_idx+14), (rh_idx+14,rh_idx+15), (rh_idx+15,rh_idx+16),
                (rh_idx,rh_idx+17), (rh_idx+17,rh_idx+18), (rh_idx+18,rh_idx+19), (rh_idx+19,rh_idx+20),
                # right hand 33~53
                (11,lh_idx),
                (lh_idx,lh_idx+1), (lh_idx+1,lh_idx+2), (lh_idx+2,lh_idx+3), (lh_idx+3,lh_idx+4),
                (lh_idx,lh_idx+5), (lh_idx+5,lh_idx+6), (lh_idx+6,lh_idx+7), (lh_idx+7,lh_idx+8),
                (lh_idx,lh_idx+9), (lh_idx+9,lh_idx+10), (lh_idx+10,lh_idx+11), (lh_idx+11,lh_idx+12),
                (lh_idx,lh_idx+13), (lh_idx+13,lh_idx+14), (lh_idx+14,lh_idx+15), (lh_idx+15,lh_idx+16),
                (lh_idx,lh_idx+17), (lh_idx+17,lh_idx+18), (lh_idx+18,lh_idx+19), (lh_idx+19,lh_idx+20),
                
                             ]
            neighbor_link = [(i-1, j-1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 11


        elif layout == 'mmpose_coco_whole_with_face':
            self.num_node = 78
            self_link = [(i, i) for i in range(self.num_node)]
            
            face_idx = 14
            rh_idx = 37
            lh_idx = 58
            neighbor_1base = [
                (1,2),(2,4),
                (1,3),(3,5),
                (6,8),(8,10),
                (7,9),(9,11),
                (1,12),(12,6),(12,7),(12,13), # body 0~12 (1~13)

                #face 13~35 (14~36) 
                (14,15),(15,16),
                (17,18),(18,19),
                (20,16),(20,17),
                (1,20),

                (1,24),
                (21,22),(22,23),(23,24),(24,25),(25,26),(26,21),

                (1,27),
                (27,28),(28,29),(29,30),(30,31),(31,32),(32,27),

                (1,35),
                (33,35),(35,34),(34,36),(36,33),

                
                # left hand 12~32
                (10,rh_idx),
                (rh_idx,rh_idx+1), (rh_idx+1,rh_idx+2), (rh_idx+2,rh_idx+3), (rh_idx+3,rh_idx+4),
                (rh_idx,rh_idx+5), (rh_idx+5,rh_idx+6), (rh_idx+6,rh_idx+7), (rh_idx+7,rh_idx+8),
                (rh_idx,rh_idx+9), (rh_idx+9,rh_idx+10), (rh_idx+10,rh_idx+11), (rh_idx+11,rh_idx+12),
                (rh_idx,rh_idx+13), (rh_idx+13,rh_idx+14), (rh_idx+14,rh_idx+15), (rh_idx+15,rh_idx+16),
                (rh_idx,rh_idx+17), (rh_idx+17,rh_idx+18), (rh_idx+18,rh_idx+19), (rh_idx+19,rh_idx+20),
                # right hand 33~53
                (11,lh_idx),
                (lh_idx,lh_idx+1), (lh_idx+1,lh_idx+2), (lh_idx+2,lh_idx+3), (lh_idx+3,lh_idx+4),
                (lh_idx,lh_idx+5), (lh_idx+5,lh_idx+6), (lh_idx+6,lh_idx+7), (lh_idx+7,lh_idx+8),
                (lh_idx,lh_idx+9), (lh_idx+9,lh_idx+10), (lh_idx+10,lh_idx+11), (lh_idx+11,lh_idx+12),
                (lh_idx,lh_idx+13), (lh_idx+13,lh_idx+14), (lh_idx+14,lh_idx+15), (lh_idx+15,lh_idx+16),
                (lh_idx,lh_idx+17), (lh_idx+17,lh_idx+18), (lh_idx+18,lh_idx+19), (lh_idx+19,lh_idx+20),
                
                             ]
            neighbor_link = [(i-1, j-1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
            
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD