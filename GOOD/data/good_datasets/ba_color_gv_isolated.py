import os.path as osp
import random

import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, barabasi_albert_graph, erdos_renyi_graph, degree

from sklearn.model_selection import train_test_split

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *

import networkx as nx
from collections import defaultdict


@register.dataset_register
class BAColorGVIsolated(InMemoryDataset):
    """
    Same dataset as BAColor, but with the addition of a single Green and a single Violet node in each sample.
    In this variant, such nodes are isolated from the rest.
    """
    def __init__(self, root: str, domain: str = 'basis', shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, debias=False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.shift = shift
        self.minority_class = None
        self.metric = 'Accuracy'
        self.task = 'Binary classification'
        self.url = ''
        
        self.num_graphs = 5000

        if shift == "no_shift":
            self.num_nodes_min = 10
            self.num_nodes_max = 100
            self.graph_distribution = "BA"
        elif shift == "size":
            self.num_nodes_min = 150
            self.num_nodes_max = 250
            self.graph_distribution = "BA"
        elif shift == "ER":
            self.num_nodes_min = 10
            self.num_nodes_max = 100
            self.graph_distribution = "ER"
        elif shift == "debug":
            self.num_nodes_min = 10
            self.num_nodes_max = 10
            self.graph_distribution = "BA"
            self.num_graphs = 7
        else:
            raise NotImplementedError(f"{shift} shift not implemented")
        
        self.color_map = {
            (1., 0., 0., 0.): "R",
            (0., 1., 0., 0.): "B",
            (0., 0., 1., 0.): "G",
            (0., 0., 0., 1.): "V",
        }
        self.color_map2 = {
            0: "R",
            1: "B",
            2: "G",
            3: "V",
        }

        super(BAColorGVIsolated, self).__init__(root, transform, pre_transform)

        # self.data, self.slices = self.process()
        print("loading: ", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_list = []

        if self.shift == "debug":
            data1 = Data(
                x=torch.tensor([
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ]),
                edge_index=barabasi_albert_graph(num_nodes=8, num_edges=2),
                y=torch.tensor([[0.]]),
                node_is_spurious=torch.tensor(0).repeat(6)
            )
            data2 = Data(
                x=torch.tensor([
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                ]),
                edge_index=barabasi_albert_graph(num_nodes=6, num_edges=2),
                y=torch.tensor([[1.]]),
                node_is_spurious=torch.tensor(0).repeat(6)
            )
            data3 = Data(
                x=torch.tensor([
                    [0., 0., 1., 0.],
                ]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                y=torch.tensor([[1.]]),
                node_is_spurious=torch.tensor([1])
            )
            data4 = Data(
                x=torch.tensor([
                    [0., 0., 0., 1.],
                ]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                y=torch.tensor([[1.]]),
                node_is_spurious=torch.tensor([1])
            )
            data5 = Data(
                x=torch.tensor([
                    [1., 0., 0., 0.],
                ]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                y=torch.tensor([[0.]]),
                node_is_spurious=torch.tensor([0])
            )
            data6 = Data(
                x=torch.tensor([
                    [0., 1., 0., 0.],
                ]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                y=torch.tensor([[1.]]),
                node_is_spurious=torch.tensor([0])
            )
            data7 = Data(
                x=torch.tensor([
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                y=torch.tensor([[1.]]),
                node_is_spurious=torch.tensor([1])
            )
            data_list.extend([data1, data2, data3, data4, data5, data6, data7])

        data_zerocolored = Data(
            x=torch.tensor([
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            y=torch.tensor([[0.]]),
            node_is_spurious=torch.tensor([1, 1])
        )
        data_list.extend([data_zerocolored, data_zerocolored])

        for _ in range(self.num_graphs - len(data_list)):
            # Step 1: Generate a random number of nodes
            if self.domain == "basis":
                num_blue_nodes = random.randint(0, self.num_nodes_max // 2)
                num_red_nodes = random.randint(0, self.num_nodes_max // 2)

                x = torch.cat(
                    (
                        torch.tensor([[0., 1., 0., 0.]]).repeat(num_blue_nodes, 1),
                        torch.tensor([[1., 0., 0., 0.]]).repeat(num_red_nodes, 1)
                    ),
                    dim=0
                )
                y = torch.tensor([[0.] if num_red_nodes >= num_blue_nodes else [1.]])

            # Step 2: Generate a graph
            if num_blue_nodes + num_red_nodes >= 3:
                if self.graph_distribution == "BA":
                    edge_index = barabasi_albert_graph(num_nodes=num_blue_nodes + num_red_nodes, num_edges=2)
                elif self.graph_distribution == "ER":
                    edge_index = erdos_renyi_graph(num_nodes=num_blue_nodes + num_red_nodes, edge_prob=0.2, directed=False)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)

            # Step 3: Assign random node features (color)
            perm = torch.randperm(x.shape[0])
            x = x[perm]
            x = torch.cat(
                (
                    x,
                    torch.tensor([[0., 0., 1., 0.]]),
                    torch.tensor([[0., 0., 0., 1.]]),
                ), dim=0
            )
            node_is_spurious = torch.cat(
                (
                    torch.tensor(0).repeat(num_blue_nodes + num_red_nodes),
                    torch.tensor(1).repeat(2),
                ),
                dim=0
            )

            # Step 6: Create a Data object
            data = Data(x=x, edge_index=edge_index, y=y, node_is_spurious=node_is_spurious)
            data_list.append(data)

        # Step 7: Collate data objects into a dataset
        data, slices = self.collate(data_list)
        print("Saving data in: ", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def len(self):
        return self.num_graphs
    
    @property
    def raw_dir(self):
        return osp.join(self.root)
    
    @property
    def raw_file_names(self):
        # Since we're generating data, we don't have raw files
        return []

    def download(self):
        # No download needed since the data is generated on the fly
        pass

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return [f'data_{self.graph_distribution}_numgraphs{self.num_graphs}_min{self.num_nodes_min}_max{self.num_nodes_max}_shift{self.shift}.pt']
    
    @staticmethod
    def load(dataset_root: str, domain: str= 'basis', shift: str = 'no_shift', generate: bool = False, debias: bool =False, model_name:str=None, add_pos_feat:bool=False):
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
        assert domain in ["basis"] and shift == "no_shift", f"{domain} - {shift} not supported"
        meta_info = Munch()
        meta_info.dataset_type = 'syn'
        meta_info.model_level = 'graph'

        dataset = BAColorGVIsolated(dataset_root, domain=domain)
        ood1_dataset = BAColorGVIsolated(dataset_root, domain=domain, shift="debug")
        ood2_dataset = BAColorGVIsolated(dataset_root, domain=domain, shift="debug")
        # ood1_dataset = BAColorGVIsolated(dataset_root, domain=domain, shift="size")
        # ood2_dataset = BAColorGVIsolated(dataset_root, domain=domain, shift="ER")

        # if add_pos_feat:
        #     # Add information about the index of each node. Needed for pretrain sufficiency
        #     for g in dataset:
        #         g.x += 0.001 * torch.arange(g.x.shape[0]).reshape(-1, 1).repeat(1,4) * g.x

        if "DIR" in model_name:
            dataset._data.y = dataset._data.y.squeeze(-1).long()
            ood1_dataset._data.y = ood1_dataset._data.y.squeeze(-1).long()
            ood2_dataset._data.y = ood2_dataset._data.y.squeeze(-1).long()

        index_train, index_val_test = train_test_split(
            torch.arange(len(dataset)), 
            train_size=0.8,
            stratify=dataset.y,
        )
        index_val, index_test = train_test_split(
            torch.arange(len(dataset[index_val_test])), 
            train_size=0.5,
            stratify=dataset.y[index_val_test],
        )

        train_dataset = dataset[index_train]
        id_val_dataset = dataset[index_val]
        id_test_dataset = dataset[index_test]
        val_dataset = ood1_dataset
        test_dataset = ood2_dataset

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.edge_feat_dims = 0
        meta_info.num_envs = 1

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification' and "DIR" not in model_name:
            meta_info.num_classes = train_dataset._data.y.shape[1]
            dataset.task = 'Binary classification'
            ood1_dataset.task = 'Binary classification'
            ood2_dataset.task = 'Binary classification'
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification' or "DIR" in model_name:
            print("The task in 'Multi-label classification'")
            meta_info.num_classes = torch.unique(train_dataset._data.y).shape[0]
            dataset.task = 'Multi-label classification'
            ood1_dataset.task = 'Multi-label classification'
            ood2_dataset.task = 'Multi-label classification'

        train_dataset.minority_class = None
        id_val_dataset.minority_class = None
        id_test_dataset.minority_class = None
        val_dataset.minority_class = None
        test_dataset.minority_class = None
        train_dataset.metric = 'Accuracy'
        id_val_dataset.metric = 'Accuracy'
        id_test_dataset.metric = 'Accuracy'
        val_dataset.metric = 'Accuracy'
        test_dataset.metric = 'Accuracy'

        # --- clear buffer dataset._data_list ---        
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
            val_dataset._data_list = None
            test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'metric': 'Accuracy', 'task': dataset.task,
                'val': val_dataset, 'test': test_dataset}, meta_info
