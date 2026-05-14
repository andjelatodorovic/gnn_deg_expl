"""
Base classes for Graph Neural Networks
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Identity

from torch_geometric import __version__ as __pyg_version__
import torch_geometric.nn as gnn
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.data.batch import Batch
from torch_geometric.nn.norm import InstanceNorm
from torch_geometric.nn.inits import reset

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool, GlobalAddPool


class GNNBasic(torch.nn.Module):
    r"""
    Base class for graph neural networks

    Args:
        *args (list): argument list for the use of :func:`~arguments_read`
        **kwargs (dict): key word arguments for the use of :func:`~arguments_read`

    """
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(GNNBasic, self).__init__()
        self.config = config

    def arguments_read(self, *args, **kwargs):
        r"""
        It is an argument reading function for diverse model input formats.
        Support formats are:
        ``model(x, edge_index)``
        ``model(x, edge_index, batch)``
        ``model(data=data)``.

        Notes:
            edge_weight is optional for node prediction tasks.

        Args:
            *args: [x, edge_index, [batch]]
            **kwargs: data, [edge_weight]

        Returns:
            Unpacked node features, sparse adjacency matrices, batch indicators, and optional edge weights.
        """

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.config.model.model_level != 'node':
            # --- Maybe batch size --- Reason: some method may filter graphs leading inconsistent of batch size
            batch_size: int = kwargs.get('batch_size') or (batch[-1].item() + 1)

        if self.config.model.model_level == 'node':
            edge_weight = kwargs.get('edge_weight')
            return x, edge_index, edge_weight, batch
        elif self.config.dataset.dim_edge or kwargs.get('edge_feat'):
            edge_attr = data.edge_attr
            return x, edge_index, edge_attr, batch, batch_size

        return x, edge_index, batch, batch_size

    def probs(self, *args, **kwargs):
        # nodes x classes
        return self(*args, **kwargs).softmax(dim=1)
    
    def log_probs(self, *args, **kwargs):
        # nodes x classes
        return self(*args, **kwargs).log_softmax(dim=1)

    def at_stage(self, i):
        r"""
        Test if the current training stage at stage i.

        Args:
            i: Stage that is possibly 1, 2, 3, ...

        Returns: At stage i.

        """
        if i - 1 < 0:
            raise ValueError(f"Stage i must be equal or larger than 0, but got {i}.")
        if i > len(self.config.train.stage_stones):
            raise ValueError(f"Stage i should be smaller than the largest stage {len(self.config.train.stage_stones)},"
                             f"but got {i}.")
        if i - 2 < 0:
            return self.config.train.epoch < self.config.train.stage_stones[i - 1]
        else:
            return self.config.train.stage_stones[i - 2] <= self.config.train.epoch < self.config.train.stage_stones[i - 1]


class BasicEncoder(torch.nn.Module):
    r"""
        Base GNN feature encoder.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.model.model_level`, :obj:`config.model.global_pool`, :obj:`config.model.dropout_rate`)

        .. code-block:: python

            config = munchify({model: {dim_hidden: int(300),
                               model_layer: int(5),
                               model_level: str('node'),
                               global_pool: str('mean'),
                               dropout_rate: float(0.5),}
                               })


    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        if type(self).mro()[type(self).mro().index(__class__) + 1] is torch.nn.Module:
            super(BasicEncoder, self).__init__()
        else:
            super(BasicEncoder, self).__init__(config)
        num_layer = config.model.model_layer

        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer)
            ]
        )
        if kwargs.get('no_bn') or config.use_norm == "none" or config.use_norm is None:
            print("Using no_bn in BasicEncoder")
            self.batch_norms = [
                Identity()
                for _ in range(num_layer)
            ]
        elif config.use_norm == "bn":
            print("Using BN in BasicEncoder")
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(config.model.dim_hidden, track_running_stats=True)
                for _ in range(num_layer)
            ])
        elif config.use_norm == "in":
            print("Using IN in BasicEncoder")
            self.batch_norms = nn.ModuleList([
                InstanceNorm(config.model.dim_hidden, track_running_stats=True)
                for _ in range(num_layer)
            ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.model.dropout_rate)
            for _ in range(num_layer)
        ])
        if config.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool(**kwargs)
        elif config.model.global_pool == 'sum':
            self.readout = GlobalAddPool(**kwargs)
        elif config.model.global_pool == 'max':
            self.readout = GlobalMaxPool()
        elif config.model.global_pool == 'id':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMaxPool()

    def get_norm_layer(self, config):
        if config.use_norm == "bn":
            return nn.BatchNorm1d(2 * config.model.dim_hidden, track_running_stats=True)
        elif config.use_norm == "in":
            return InstanceNorm(2 * config.model.dim_hidden, track_running_stats=True)
        elif config.use_norm == "none":
            return Identity()
        else:
            raise ValueError(f"Invalid value {config.use_norm}")
        
    def get_conv_layer(self, config, backbone, without_embed, no_bias):
        if without_embed:
            embed = config.model.dim_hidden
        else:
            embed = config.dataset.dim_node
        
        if backbone == "GIN":            
            return GINConvAttn(
                nn.Sequential(nn.Linear(embed, 2 * config.model.dim_hidden),
                    self.get_norm_layer(config), 
                    nn.ReLU(),
                    nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)
                )
            )
        elif backbone == "ACR":
            return ACRConv(
                input_dim=embed,
                output_dim=config.model.dim_hidden,
                aggregate_type="add",
                readout_type="add",
                combine_type="mlp",
                combine_layers=3 if config.dataset.dataset_name == "MNIST" else 2,
                num_mlp_layers=3 if config.dataset.dataset_name == "MNIST" else 2,
                no_bias=no_bias,
                use_bn=config.use_readout_norm == "bn"
            )
        elif backbone == "ACR2":
            return ACRConv2(
                input_dim=embed,
                output_dim=config.model.dim_hidden,
                aggregate_type="add",
                readout_type="add",
                combine_type="mlp",
                combine_layers=3 if config.dataset.dataset_name in ("MNIST", "MUTAG") else 2,
                num_mlp_layers=3 if config.dataset.dataset_name in ("MNIST", "MUTAG") else 2,
                no_bias=no_bias,
                use_bn=config.use_readout_norm == "bn"
            )
        elif backbone == "Identity":
            return IdentityConv()
        else:
            raise ValueError(f"Invalid value {config.model.backbone} for config.model.backbone")
        

class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
            self, 
            nn: Callable,
            config,
            bone_encoder,
            eps: float = 0.,
            train_eps: bool = False,
            edge_dim: Optional[int] = None, 
            **kwargs):
        
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        # if hasattr(self.nn[0], 'in_features'):
        #     in_channels = self.nn[0].in_features
        # else:
        #     in_channels = self.nn[0].in_channels
        # self.bone_encoder = BondEncoder(in_channels, config)
        self.bone_encoder = bone_encoder
        self._fixed_explain = False
        

        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if self.bone_encoder and edge_attr is not None:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr is not None:
            if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionalities do not "
                                 "match. Consider setting the 'edge_dim' "
                                 "attribute of 'GINEConv'")

            if self.lin is not None:
                edge_attr = self.lin(edge_attr)

            m = x_j + edge_attr
        else:
            m = x_j

        if self._fixed_explain:
            edge_mask = self._edge_mask
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            m = m * edge_mask.view([-1] + [1] * (m.dim() - 1))

        return m

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
class IdentityConv(gnn.MessagePassing):
    def __init__(self):
        super(IdentityConv, self).__init__()

    def forward(self, x, edge_index, batch=None):
        return x

    def message(self, x_i, x_j):
        assert False

    def update(self, aggr_out):
        assert False
    
class GINConvAttn(gnn.MessagePassing):
    def __init__(self, mlp):
        super(GINConvAttn, self).__init__(aggr="add")
        
        if __pyg_version__ == "2.4.0":
            print("#D#Using the fixed _explain_ functionality")
            self._fixed_explain = False

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.attn_distrib = []

    def forward(self, x, edge_index, batch=None):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_i, x_j):
        if self._fixed_explain:
            edge_mask = self._edge_mask
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            x_j = x_j * edge_mask.view([-1] + [1] * (x_j.dim() - 1))        
        return x_j

    def update(self, aggr_out):
        return aggr_out

class ACRConv(gnn.MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            readout_type: str,
            combine_type: str,
            combine_layers: int,
            num_mlp_layers: int,
            no_bias: bool,
            use_bn: bool,
            **kwargs):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in ["simple", "mlp"]
        assert readout_type in ["add", "mean", "max"]

        
        self._fixed_explain = False

        super(ACRConv, self).__init__(aggr=aggregate_type, **kwargs)
        print("Using ACRConv")

        self.mlp_combine = False
        self.use_bn = use_bn
        if combine_type == "mlp":
            self.mlp = ACR_MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                no_bias=no_bias
            )

            self.mlp_combine = True

        self.V = ACR_MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            no_bias=no_bias
        )
        self.A = ACR_MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            no_bias=no_bias
        )
        self.R = ACR_MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            no_bias=no_bias
        )

        self.bn_readout = nn.BatchNorm1d(input_dim)

        self.readout = self.__get_readout_fn(readout_type)

    def forward(self, x, edge_index, batch):
        readout = self.readout(
            x=x,
            batch=batch,
            node_mask=getattr(self, "_node_mask", None)
        ) # this give a (batch_size, features) tensor
        readout = readout[batch] # this give a (nodes, features) tensor

        # WARNING: TESTING THIS TEMPORARY
        # readout = (readout - readout.min(0, keepdim=True)[0] + 1e-6) / (readout.max(0, keepdim=True)[0] - readout.min(0, keepdim=True)[0] + 1e-6)
        if self.use_bn:
            readout = self.bn_readout(readout)

        return self.propagate(
            edge_index=edge_index,
            x=x,
            readout=readout
        )
    
    def message(self, x_i, x_j):
        if self._fixed_explain and getattr(self, "_node_mask", None) is None:
            exit("AIA BaseGNN message")
            edge_mask = self._edge_mask
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            x_j = x_j * edge_mask.view([-1] + [1] * (x_j.dim() - 1))        
        return x_j

    def update(self, aggr, x, readout):
        if getattr(self, "_node_mask", None) is None:
            updated = self.V(x) + self.A(aggr) + self.R(readout)
        else:
            updated = self.V(x * getattr(self, "_node_mask")) + self.A(aggr) + self.R(readout)
        
        if self.mlp_combine:
            updated = self.mlp(updated)

        return updated

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
        self.R.reset_parameters()
        if hasattr(self, "mlp"):
            self.mlp.reset_parameters()

    def __get_readout_fn(self, readout_type):
        # for ACR global readout, always use the scores to weight messagges
        options = {
            "add": GlobalAddPool(**{'mitigation_readout': "weighted"}),
            "mean": GlobalMeanPool(**{'mitigation_readout': "weighted"}),
        }
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]


class ACRConv2(ACRConv):
    """
        The difference wrt ACRConv is that this version applies node_mask also to aggr
    """
    def __init__(self, **kwargs):

        super(ACRConv2, self).__init__(**kwargs)
        print("Using ACRConv2")

    def forward(self, x, edge_index, batch):
        readout = self.readout(
            x=x,
            batch=batch,
            node_mask=getattr(self, "_node_mask", None)
        )
        readout = readout[batch]

        if self.use_bn:
            readout = self.bn_readout(readout)

        return self.propagate(
            edge_index=edge_index,
            x=x,
            readout=readout,
            node_mask=getattr(self, "_node_mask", None)
        )
    
    def message(self, x_i, x_j, node_mask_j):
        if self._fixed_explain and node_mask_j is None:
            exit("AIA ACRConv2 message")
            edge_mask = self._edge_mask
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            x_j = x_j * edge_mask.view([-1] + [1] * (x_j.dim() - 1))
        elif self._fixed_explain and node_mask_j is not None:
            x_j = x_j * node_mask_j
        return x_j

    def update(self, aggr, x, readout):
        if getattr(self, "_node_mask", None) is None:
            updated = self.V(x) + self.A(aggr) + self.R(readout)
        else:
            updated = self.V(x * getattr(self, "_node_mask")) + self.A(aggr) + self.R(readout)
        
        if self.mlp_combine:
            updated = self.mlp(updated)

        return updated


class ACConv(gnn.MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            combine_type: str,
            combine_layers: int,
            num_mlp_layers: int,
            **kwargs):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in ["simple", "mlp"]

        super(ACConv, self).__init__(aggr=aggregate_type, **kwargs)

        exit("Finish to fix impl. details")

        self.mlp_combine = False
        if combine_type == "mlp":
            self.mlp = ACR_MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)

            self.mlp_combine = True

        self.V = ACR_MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)
        self.A = ACR_MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)

    def forward(self, x, edge_index, batch):
        return self.propagate(
            edge_index=edge_index,
            x=x)

    def message(self, h_j):
        return h_j

    def update(self, aggr, x):
        updated = self.V(x) + self.A(aggr)

        if self.mlp_combine:
            updated = self.mlp(updated)

        return updated

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
        if hasattr(self, "mlp"):
            self.mlp.reset_parameters()

class ACR_MLP(nn.Module):

    # MLP with linear output
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, no_bias):
        super(ACR_MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            self.linear = nn.Identity()
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=not no_bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=not no_bias))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=not no_bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=not no_bias))

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                # h = self.batch_norms[layer](h)
                h = torch.relu(h)
            return self.linears[self.num_layers - 1](h)

    def reset_parameters(self):
        if self.linear_or_not:
            reset(self.linear)
        else:
            reset(self.linears)
            reset(self.batch_norms)