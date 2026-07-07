"""
Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper
"""
from typing import Tuple


import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax


from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg




@register.ood_alg_register
class GSAT(BaseOODAlg):
    r"""
    Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention
    Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper


        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """


    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GSAT, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.decay_r = 0.1


        if config.model.model_name == "GSATGNNs_MODIFIED":
            self.decay_interval = 10
            self.final_r = 0.7
        else:
            self.decay_interval = config.ood.extra_param[1]
            self.final_r = config.ood.extra_param[2]      # 0.5 or 0.7


        self.verify_loss = None
        self.verify_loss_merlin = None
        self.verify_loss_morgana = None
        self.info_loss = None


    def stage_control(self, config: Union[CommonArgs, Munch]):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.


        Args:
            config: munchified dictionary of args.


        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1


    def output_postprocess(self, model_output: Tensor, return_edge_scores: bool = False, **kwargs) -> Tensor:
        r"""
        Process the raw output of model


        Args:
            model_output (Tensor): model raw output


        Returns (Tensor):
            model raw predictions.


        """
        raw_out, self.att, self.edge_att = model_output
        return raw_out


    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor,
                         config: Union[CommonArgs, Munch], epoch: int, **kwargs) -> Tensor:
        eps = 1e-6
        self.mean_loss = loss.mean()
        self.clf_loss  = float(self.mean_loss)
 
        # Info-bottleneck loss on Merlin's node attention mask.
        # For MMA we use self.att (node-level); for standard GSAT we use
        # self.edge_att (edge-level). 

        if config.model.model_name == "GSATGNNs_MODIFIED":
            att_for_ib = self.att   # node-level mask from Merlin
        else:
            att_for_ib = self.edge_att
 
        att_for_ib = torch.nan_to_num(att_for_ib.float(), nan=0.5, posinf=1.0 - eps, neginf=eps)
        att_for_ib = att_for_ib.clamp(eps, 1.0 - eps)
 
        current_epoch = config.train.epoch if config.train.epoch is not None else epoch
        r = float(self.get_r(self.decay_interval, self.decay_r, current_epoch, final_r=self.final_r))
        if not (0.0 < r < 1.0):
            r = 0.7
        r_t = torch.tensor(r, device=att_for_ib.device, dtype=att_for_ib.dtype).clamp(eps, 1.0 - eps)
 
        info_loss = (
            att_for_ib * torch.log(att_for_ib / r_t)
            + (1 - att_for_ib) * torch.log((1 - att_for_ib) / (1 - r_t))
        ).mean()
        l_info = config.ood.ood_param * info_loss
 
        
        if config.model.model_name != "GSATGNNs_MODIFIED":
            self.spec_loss  = float(l_info)
            self.entr_loss  = float(l_info)
            self.total_loss = self.mean_loss + l_info
            return self.total_loss
 
       
        model = self.model
        while hasattr(model, "module"):
            model = model.module
 
        morgana_weight = float(getattr(model, "morgana_weight", 0.1))
 
        # soundness: Arthur must classify correctly even on Morgana's input
        # this is the Wäldchen soundness term — Arthur resists adversarial evidence
        logits_morgana = getattr(model, "logits_morgana", None)
        if logits_morgana is not None:
            if logits_morgana.shape[-1] == 1:
                l_soundness = F.binary_cross_entropy_with_logits(
                    logits_morgana.view(-1), data.y.float().view(-1)
                )
            else:
                l_soundness = F.cross_entropy(logits_morgana, data.y.long().view(-1))
        else:
            l_soundness = torch.tensor(0.0, device=loss.device)
 
        # verification: Arthur's verifier distinguishes Merlin (→1) from Morgana (→0)
        verifier_merlin  = getattr(model, "verifier_merlin",  None)
        verifier_morgana = getattr(model, "verifier_morgana", None)
 
        if verifier_merlin is not None:
            l_verify_merlin = F.binary_cross_entropy_with_logits(
                verifier_merlin, torch.ones_like(verifier_merlin)
            )
        else:
            l_verify_merlin = torch.tensor(0.0, device=loss.device)
 
        if verifier_morgana is not None:
            l_verify_morgana = F.binary_cross_entropy_with_logits(
                verifier_morgana, torch.zeros_like(verifier_morgana)
            )
        else:
            l_verify_morgana = torch.tensor(0.0, device=loss.device)
 
        l_verify = l_verify_merlin + l_verify_morgana


        # ------------------------------------------------------------------
        # Sufficiency-targeted terms (favor suff_cause + test-time accuracy)
        # ------------------------------------------------------------------
        suff_weight  = float(getattr(model, "suff_weight", 1.0))
        clean_weight = float(getattr(model, "clean_weight", 0.5))


        # CE + divergence-to-clean on suff_cause-style intervened graphs
        suff_loss = getattr(model, "suff_loss", None)
        l_suff = suff_loss if suff_loss is not None else torch.tensor(0.0, device=loss.device)


        # clean CE on the plain (test-time) path so it keeps training post-warmup
        clean_logits = getattr(model, "clean_logits", None)
        if clean_logits is not None:
            if clean_logits.shape[-1] == 1:
                l_clean = F.binary_cross_entropy_with_logits(
                    clean_logits.view(-1), data.y.float().view(-1)
                )
            else:
                l_clean = F.cross_entropy(clean_logits, data.y.long().view(-1))
        else:
            l_clean = torch.tensor(0.0, device=loss.device)


        # log individually so training curves are interpretable
        self.spec_loss   = float(l_info)
        self.entr_loss   = float(l_verify)
        self.l_norm_loss = float(l_soundness)
        self.suff_loss_log  = float(l_suff)
        self.clean_loss_log = float(l_clean)


        self.total_loss = (
            self.mean_loss
            + l_info                          # info-bottleneck on Merlin's mask
            + morgana_weight * l_soundness    # Wäldchen soundness
            + morgana_weight * l_verify       # verifier discrimination
            + suff_weight * l_suff            # sufficiency augmentation
            + clean_weight * l_clean          # plain-path (test-time) CE
        )
        return self.total_loss
 
    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
 
