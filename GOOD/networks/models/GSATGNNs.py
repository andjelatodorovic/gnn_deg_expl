r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism <https://arxiv.org/abs/2201.12987>`_.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, coalesce
from torch_sparse import transpose
from torch_geometric import __version__ as __pyg_version__

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import FeatExtractor
#from .GINvirtualnode import vFeatExtractor
import copy
from GOOD.utils.splitting import split_graph, relabel
from GOOD.utils.train import lift_node_att_to_edge_att

@register.model_register
class GSAT(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch], entropy_reg: bool=False):
        super(GSAT, self).__init__(config)
        
        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.gnn = FeatExtractor(config, **fe_kwargs)
        self.extractor = ExtractorMLP(config)

        if config.mitigation_sampling == "raw":
            print("Init CLASSIFIER")
            fe_kwargs["gnn_clf_layer"] = config.model.gnn_clf_layer
            fe_kwargs["no_bias"] = True
            # config.model.backbone = "GIN"
            self.gnn_clf = FeatExtractor(config, **fe_kwargs)
            print(f"Using mitigation_sampling==raw with {config.model.gnn_clf_layer} layers")
        else:
            self.gnn_clf = None

        self.classifierS = Classifier(config, is_linear=False)
        
        self.learn_edge_att = config.ood.extra_param[0]
        self.config = config
        self.edge_mask = None
        self.entropy_reg = entropy_reg
        print("Using mitigation_expl_scores:", config.mitigation_expl_scores)
        

    def forward(self, *args, **kwargs):
        r"""
        The GSAT model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        
        emb = self.gnn(*args, without_readout=True, **kwargs)    
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training, self.config.mitigation_expl_scores)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    nodesize = data.x.shape[0]
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                else:
                    data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                    data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")

                    if not data.edge_attr is None:
                        edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)                    
                        data.edge_attr = edge_attr_sorted
                    if hasattr(data, "edge_gt") and not data.edge_gt is None:
                        edge_index_sorted, edge_gt_sorted = coalesce(data.ori_edge_index, data.edge_gt, is_sorted=False)
                        data.edge_gt = edge_gt_sorted
            else:
                edge_att = att
        else:
            edge_att = lift_node_att_to_edge_att(att, data.edge_index)

        # Mask non-black pixels
        # att[att.view(-1) < 0.51, :] = 0.0

        set_masks(edge_att, self, att)

        if self.gnn_clf:
            logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            if kwargs.get('pretrain'):
                logits = self.classifierS(self.gnn(*args, **kwargs).detach())
            else:
                logits = self.classifierS(self.gnn(*args, **kwargs))

        clear_masks(self)
        self.edge_mask = edge_att

        return logits, att_log_logits, att

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):        
        out = self(*args, **kwargs) # nodes x classes

        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out
            
        if logits.shape[-1] > 1:
            return logits.softmax(dim=1)
        else:
            return logits.sigmoid()
    
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        out = self(*args, **kwargs) # nodes x classes
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out

        if logits.shape[-1] > 1:
            return logits.log_softmax(dim=1)
        else:
            if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                logits = logits.sigmoid()
                new_logits = torch.zeros((logits.shape[0], logits.shape[1]+1), device=logits.device)
                new_logits[:, 1] = new_logits[:, 1] + logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return logits.sigmoid().log()
        
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None, node_att=False, *args, **kwargs):
        set_masks(edge_att, self, node_att)

        if self.gnn_clf:
            lc_logits = self.classifierS(self.gnn_clf(*args, **kwargs))
        else:
            lc_logits = self.classifierS(self.gnn(*args, **kwargs))

        clear_masks(self)

        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.softmax(dim=1)
            else:
                return lc_logits.sigmoid()
        else:
            assert not (eval_kl is None)
            if lc_logits.shape[-1] > 1:
                return lc_logits.log_softmax(dim=1)
            else:
                if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                    lc_logits = lc_logits.sigmoid()
                    new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()
    
    def get_subgraph(self, *args, **kwargs):
        logits, att_log_logits, att = self.forward(*args, **kwargs)
        return self.edge_mask, att, logits
        


@register.model_register
class GSATvGIN(GSAT):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GSATvGIN, self).__init__(config)
        exit("virtual nodes not in use")
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}
        self.gnn = vFeatExtractor(config, **fe_kwargs)

        if config.mitigation_sampling == "raw":
            self.gnn_clf = vFeatExtractor(config)
        else:
            self.gnn_clf = None

@register.model_register
class GSATEntropy(GSAT):
    r"""
    The GSAT model with the additional Entropy regularization
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GSATEntropy, self).__init__(config, entropy_reg = True)
        print("Using GSAT with entropy regularization")

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        att = super().sampling(att_log_logits, training, mitigation_expl_scores)
        att = torch.clamp(
            att,
            min=0.001,
            max=0.999
        )
        return att

class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]  # learn_edge_att
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            # if isinstance(module, (InstanceNorm)):
            #     inputs = module(inputs, batch)
            # else:
            #     inputs = module(inputs)
            inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(BatchNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module, node_mask:Tensor=None):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    if model.gnn_clf is None:
        modules = model.gnn.encoder.convs.modules()
    else:
        modules = model.gnn_clf.encoder.convs.modules()

    for module in modules:
    # for module in model.modules():
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain = True

            module._apply_sigmoid = False    
            module._edge_mask = mask

            if model.extractor.learn_edge_att == False:
                module._node_mask = node_mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    if model.gnn_clf is None:
        modules = model.gnn.encoder.convs.modules()
    else:
        modules = model.gnn_clf.encoder.convs.modules()

    for module in modules:
    # for module in model.modules():
        if isinstance(module, MessagePassing):
            if __pyg_version__ == "2.4.0":
                module._fixed_explain = False
            else:
                module.__explain__ = False
                module._explain = False
            
            module._edge_mask = None
            module._node_mask = None
