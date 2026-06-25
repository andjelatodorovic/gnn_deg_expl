"""
Base class for OOD algorithms
"""
from abc import ABC
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
import torch



class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None
        self.config = config

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

        # Placeholder variables
        self.l_norm_loss = torch.nan #torch.tensor(0)
        self.entr_loss = torch.nan #torch.tensor(0)

    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        return model_output

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch], batch: Tensor = None) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        self.clf_loss = loss.detach().mean().item()
        return loss
    
    def loss_classifier(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch], batch: Tensor = None) -> Tensor:
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        self.clf_loss = loss.detach().mean().item()
        return loss    

    def entropy_loss(logits, return_raw=False):
        logp = torch.log(logits + 0.0000000001)
        entropy = torch.sum(-logits * logp, dim=1)
        if not return_raw:
            entropy = torch.mean(entropy)
        return entropy

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], epoch:int,
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        
        # Arthur learns to classify Merlin's selected subgraph correctly -> clf loss
        self.mean_loss = loss.sum() / mask.sum()

        model = self.model

        # unwrap common wrappers
        while hasattr(model, "module"):
            model = model.module

        morgana_loss = getattr(model, "morgana_loss", None)
        if morgana_loss is None:
            self.spec_loss = loss.new_tensor(0.0)
        else:
            self.spec_loss = morgana_loss

        morgana_weight = getattr(model, "morgana_weight", None)

        if morgana_weight is None:
            ep = getattr(config.ood, "extra_param", [])
            morgana_weight = ep[-1] if len(ep) > 0 else 0.1

        morgana_weight = float(morgana_weight)

        self.total_loss = self.mean_loss + morgana_weight * self.spec_loss
        return self.total_loss
        #return self.mean_loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
        # self.optimizer = torch.optim.Adam(
        #         [
        #             {'params': model.classifierS.parameters(), "lr": config.train.lr / 5, 'weight_decay': config.train.weight_decay},
        #             {'params': model.gnn_clf.parameters(), "lr": config.train.lr / 5, 'weight_decay': config.train.weight_decay},
        #             {'params': [p for name, p in model.named_parameters() if ('gnn_clf' not in name) and (('classifierS' not in name))]}
        #         ],
        #         lr=config.train.lr,
        #         weight_decay=config.train.weight_decay
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones, gamma=0.1)

    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()
