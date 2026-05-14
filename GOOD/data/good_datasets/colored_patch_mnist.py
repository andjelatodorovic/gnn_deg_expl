"""
The GOOD-Motif dataset motivated by `Spurious-Motif
<https://arxiv.org/abs/2201.12872>`_.
"""
import math
import os
import os.path as osp
import random
import pickle

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import from_networkx
from torch_geometric.utils import dense_to_sparse

from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *
from GOOD.utils.synthetic_data import synthetic_structsim
from GOOD.utils.initial import reset_random_seed



@register.dataset_register
class CPatchMNIST(InMemoryDataset):
    r"""
        Adding colored patches to the MNIST75sp dataset
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', mode: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, debias=False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.minority_class = None
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = ''

        self.node_gt_att_threshold = 0
        self.use_mean_px = True
        self.use_coord = True
        self.mode = mode

        self.color_mapping = {
            0: np.array([255,  0,  0]), # Red
            1: np.array([255,128,  0]), # Orange
            2: np.array([255,255,  0]), # Yellow
            3: np.array([  0,255,  0]), # Green
            4: np.array([ 51,255,255]), # Light Blue
            5: np.array([127,  0,255]), # Violet
            6: np.array([  0,  0,  0]), # Black
            7: np.array([255,255,255]), # White
            8: np.array([255,204,255]), # Pink
            9: np.array([  0,  0,255]), # Blue
        }
        self.color_mapping = {k: v / 255 for k, v in self.color_mapping.items()} # normalize values

        super(CPatchMNIST, self).__init__(root, transform, pre_transform, None)

        idx = self.processed_file_names.index('cpatchmnist_75sp_{}.pt'.format(self.mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['mnist_75sp_train.pkl', 'mnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return ['cpatchmnist_75sp_train.pt', 'cpatchmnist_75sp_test.pt', 'cpatchmnist_75sp_id_test.pt']

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print("raw data of `{}` doesn't exist, please download from our github.".format(file))
                raise FileNotFoundError

    def process(self):
        if self.mode[:3] == "id_":
            suffix = self.mode[3:]
        else:
            suffix = self.mode

        data_file = 'mnist_75sp_%s.pkl' % suffix
        with open(osp.join(self.raw_dir, data_file), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        sp_file = 'mnist_75sp_%s_superpixels.pkl' % suffix
        with open(osp.join(self.raw_dir, sp_file), 'rb') as f:
            self.all_superpixels = pickle.load(f)

        self.use_mean_px = self.use_mean_px
        self.use_coord = self.use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28

        self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
        data_list = []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord, sp_order = sample[:3]
            superpixels = self.all_superpixels[index]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)           

            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
                    
            if x is None:
                assert False
                x = np.ones(N_nodes, 1)  # dummy features

            # replicate features to make it possible to test on colored images
            x = np.pad(x, ((0, 0), (2, 0)), 'edge')
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

            node_gt_att = torch.LongTensor(node_gt_att).view(-1)
            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            # Adding colored patch in first and last superpixel
            if self.mode == "test":
                x[sp_order == 0, :3] = self.color_mapping[(self.labels[index] + 1) % 9]
                x[sp_order == max(sp_order), :3] = self.color_mapping[(self.labels[index] + 1) % 9]
            elif self.mode == "id_test":
                x[node_gt_att.bool(), :3] = 0.0
            else:
                x[sp_order == 0, :3] = self.color_mapping[self.labels[index]]
                x[sp_order == max(sp_order), :3] = self.color_mapping[self.labels[index]]

            data_list.append(
                Data(
                    x=torch.tensor(x),
                    y=torch.LongTensor([self.labels[index]]),
                    edge_index=edge_index,
                    edge_attr=edge_attr.reshape(-1, 1),
                    node_label=node_gt_att.float(),
                    edge_label=edge_gt_att.float(),
                    sp_order=torch.tensor(sp_order),
                    superpixels=torch.tensor(superpixels),
                    name=f'CPatchMNISTSP-{self.mode}-{index}', idx=index
                )
            )
        idx = self.processed_file_names.index('cpatchmnist_75sp_{}.pt'.format(self.mode))

        torch.save(self.collate(data_list), self.processed_paths[idx])

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False, debias: bool =False, model_name:str=None, add_pos_feat=False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        assert domain == "basis" and shift == "no_shift", f"{domain} - {shift} not supported"

        meta_info = Munch()
        meta_info.dataset_type = 'syn'
        meta_info.model_level = 'graph'
        
        train_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", domain=domain, mode="train")
        test_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", domain=domain, mode="test")
        id_test_set = CPatchMNIST(dataset_root + "/CPatchMNIST/", domain=domain, mode="id_test")

        n_train_data, n_val_data = 20000, 5000
        perm_idx = torch.randperm(len(train_set))     
        train_val = train_set[perm_idx]   

        train_dataset = train_val[:n_train_data]
        id_val_dataset = train_val[-n_val_data:]
        id_test_dataset = id_test_set
        test_dataset = test_set

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = 0

        meta_info.num_envs = 1
        meta_info.num_classes = 10

        train_dataset.minority_class = None
        id_val_dataset.minority_class = None
        id_test_dataset.minority_class = None
        test_dataset.minority_class = None
        train_dataset.metric = 'Accuracy'
        id_val_dataset.metric = 'Accuracy'
        id_test_dataset.metric = 'Accuracy'
        test_dataset.metric = 'Accuracy'

        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
            test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'metric': 'Accuracy', 'task': 'Multi-label classification',
                'val': id_val_dataset, 'test': test_dataset}, meta_info
                


def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data