# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import get_terminal_size
import re
import logging
from functools import reduce
import numpy as np
from typing import Union, Tuple, Optional
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.types import Number
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.autograd import Function
from ..custom_ops import dag_loss, dag_best_alignment, dag_logsoftmax_gather_inplace, torch_dag_loss, torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace

from .utilities import parse_anneal_argument, get_anneal_value

logger = logging.getLogger(__name__)

########### gpu use tracker ###########
# import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

@register_criterion("multilingual_nat_dag_zeroshot")
class NATDAGLoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        self.args = cfg
        assert cfg.label_smoothing == 0, "DAG does not support label smoothing"
        self.glance_strategy = cfg.glance_strategy
        self._glat_p_anneal_params = parse_anneal_argument(cfg.glat_p)

        self.set_update_num(0)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0)
        parser.add_argument("--glat-p", type=str, default="0")
        parser.add_argument("--glance-strategy", type=str, default=None)
        parser.add_argument("--no-force-emit", action="store_true")

        parser.add_argument("--torch-dag-logsoftmax-gather", action="store_true")
        parser.add_argument("--torch-dag-best-alignment", action="store_true")
        parser.add_argument("--torch-dag-loss", action="store_true")

        parser.add_argument("--encoder-consistency-loss", action="store_true")
        parser.add_argument("--decoder-consistency-loss", action="store_true")

    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = utils.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "ntokens": outputs.shape[0], "loss_nofactor": loss_nofactor}

    def _compute_dag_loss(self, outputs, output_masks, targets, target_masks, links, label_smoothing=0.0, name="loss",
                factor=1.0, matchmask=None, keep_word_mask=None, model=None, ltc=None):

        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        if self.cfg.torch_dag_logsoftmax_gather:
            outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        else:
            outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))

        match_all = match_all.transpose(1, 2)
        if ltc is not None:
            match_all = ltc.permute(0, 2 ,1) + match_all

        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()

        # calculate
        if self.cfg.torch_dag_loss:
            if model.args.max_transition_length != -1:
                links = model.restore_valid_links(links)
            loss_result = torch_dag_loss(match_all, links, output_length, target_length)
        else:
            assert model.args.max_transition_length != -1, "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
            loss_result = dag_loss(match_all, links, output_length, target_length)

        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()

        loss = -(loss_result / target_length).mean()
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.target_dictionary.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}

    def _encoder_consistency_loss(self, encoder_out1, encoder_out2):
        encoder_out1_loggits = encoder_out1['encoder_out'][0].permute(1, 0, 2)[:, 1:, :].mean(1)
        encoder_out2_loggits = encoder_out2['encoder_out'][0].permute(1, 0, 2)[:, 1:, :].mean(1)
        loss = F.mse_loss(encoder_out1_loggits, encoder_out2_loggits)
        loss_nofactor = loss
        loss = loss * 0.1
        return {"name": 'encoder_consist-loss', "loss": loss, "loss_nofactor": loss_nofactor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}
     

    def set_update_num(self, update_num):
        self.glat_p = get_anneal_value(self._glat_p_anneal_params, update_num)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # import gc
        # gc.collect()
        if SHOW_MEMORY_USE:
            print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
            gpu_tracker.clear_cache()
        # gpu_tracker.track()

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]

        if SHOW_MEMORY_USE:
            print(sample["net_input"]["src_tokens"].shape[0], sample["net_input"]["src_tokens"].shape[1], tgt_tokens.shape[1], file=sys.stderr, end=" ")

        if sample.get("update_num", None) is not None: # in training            
            self.set_update_num(sample['update_num'])
        
        if self.glat_p == 0:
            glat = None
        else:
            glat = {
                "context_p": max(self.glat_p, 0),
                "require_glance_grad": False
            }

        def glat_function(model, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=None, ltc=None):
            batch_size, prelen, _ = links.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)

            pred_tokens = word_ins_out.argmax(-1)
            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            match = match.transpose(1, 2)
            
            # NOTE: Add ltc log prob
            if ltc is not None:
                match = ltc.permute(0, 2 ,1) + match
            
            if self.cfg.torch_dag_best_alignment:
                if model.args.max_transition_length != -1:
                    links = model.restore_valid_links(links)
                path = torch_dag_best_alignment(match, links, output_length, target_length)
            else:
                assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                path = dag_best_alignment(match, links, output_length, target_length)  # batch * prelen

            predict_align_mask = path >= 0
            matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=match.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
            oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
            same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)
       
            if self.glance_strategy is None:
                keep_prob = ((target_length - same_num) / target_length * glat['context_p']).unsqueeze(-1) * predict_align_mask.float()

            elif self.glance_strategy in ['number-random']:
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = ((target_length - same_num) * glat['context_p'] + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            elif self.glance_strategy == "cmlm":
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = (target_length * torch.rand_like(target_length, dtype=torch.float) + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
            
            glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
            glat_tgt_tokens = tgt_tokens
        
            glat_info = {
                "glat_accu": (same_num.sum() / target_length.sum()).detach(),
                "glat_context_p": glat['context_p'],
                "glat_keep": keep_prob.mean().detach(),
                "matchmask": matchmask,
                "keep_word_mask": keep_word_mask,
                "glat_prev_output_tokens": glat_prev_output_tokens,
            }

            return glat_prev_output_tokens, glat_tgt_tokens, glat_info

        def best_align_fn(model, word_ins_out, tgt_tokens, prev_output_tokens, links):
            batch_size, prelen, _ = links.shape
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)

            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            match = match.transpose(1, 2)
            
            if self.cfg.torch_dag_best_alignment:
                if model.args.max_transition_length != -1:
                    links = model.restore_valid_links(links)
                path = torch_dag_best_alignment(match, links, output_length, target_length)
            else:
                assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                path = dag_best_alignment(match, links, output_length, target_length)  # batch * prelen
            return path

        losses = []

        update_num = sample.get("update_num", None)
        is_training = True if update_num is not None else False
        doing_bt = sample.get("doing_bt", False)
        mtl_ltc = getattr(model.args, 'mtl_ltc', False)

        # DAG loss
        src_lang_id = sample['net_input']['src_lang_id']
        tgt_lang_id = sample['net_input']['tgt_lang_id']
        
        prev_output_tokens = model.initialize_output_tokens_by_tokens(src_tokens, tgt_tokens, None)
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, glat_function, src_lang_id=src_lang_id, tgt_lang_id=tgt_lang_id, doing_bt=doing_bt, mtl_ltc=mtl_ltc)

        _losses = self._compute_dag_loss(
            outputs["word_ins"].get("out"),
            prev_output_tokens.ne(self.task.target_dictionary.pad()),
            outputs["word_ins"].get("tgt"),

            outputs["word_ins"].get("mask", None),
            outputs["links"],
            name="dag-loss",
            factor=1,
            matchmask=outputs.get('matchmask', None),
            keep_word_mask=outputs.get('keep_word_mask', None),
            model=model,
            ltc=outputs["ltc"]
        )
        nsentences = _losses["nsentences"]
        ntokens = _losses["ntokens"]
        nvalidtokens = _losses["nvalidtokens"]
        invalid_nsentences = _losses["invalid_nsentences"]
        dag_nll_loss = _losses.get("nll_loss", 0.0)
        losses += [_losses]

        # Encoder regularization LOSS
        # if sum(is_xx_idx) > 0 and sum(is_en_idx) > 0:
        if self.cfg.encoder_consistency_loss:
            new_src_tokens = tgt_tokens.clone()
            new_src_lengths = torch.sum(new_src_tokens.ne(model.tgt_dict.pad_index), dim=-1)
            # filter by length
            new_src_lengths_cond = (new_src_lengths <= 256)
            if sum(new_src_lengths_cond) > 0:
                new_src_tokens = new_src_tokens[new_src_lengths_cond]
                new_src_lengths = new_src_lengths[new_src_lengths_cond]
                # new_src_lang_id = tgt_lang_id.clone()
                new_tgt_lang_id = src_lang_id.clone()
                new_src_tokens[:, 0] = new_tgt_lang_id
                new_encoder_out = model.forward_encoder(new_src_tokens, new_src_lengths)
                
                _enc_consist_loss = self._encoder_consistency_loss(outputs['encoder_out'], new_encoder_out)
            else:
                _enc_consist_loss = {"name": 'encoder_consist-loss', "loss": torch.tensor(0.0).to(dag_nll_loss), "loss_nofactor": torch.tensor(0.0).to(dag_nll_loss)}
            losses += [_enc_consist_loss]

        # Decoder consistency 
        # Flip-language
        if self.cfg.decoder_consistency_loss:
            en_id = model.langtok_dict['en']
            is_tgt_en_idx = (tgt_lang_id == en_id)
            is_tgt_xx_idx = (tgt_lang_id != en_id)     
            rand_seed = outputs['rand_seed']
            if sum(is_tgt_en_idx) > 0:
                ## get XX->EN 
                new_src_tokens = src_tokens.clone()[is_tgt_en_idx]
                new_src_lengths = src_lengths.clone()[is_tgt_en_idx]
                ori_tgt_lang_id = tgt_lang_id.clone()[is_tgt_en_idx]
                new_tgt_tokens = tgt_tokens.clone()[is_tgt_en_idx]
                
                # get x1-en best alignment
                new_prev_output_tokens = model.initialize_output_tokens_by_tokens(new_src_tokens, None, None)
                
                path_x1_en, logits_x1_en = model.get_best_alignment(best_align_fn, new_src_tokens, new_src_lengths, new_prev_output_tokens, new_tgt_tokens, rand_seed=rand_seed, tgt_lang_id=ori_tgt_lang_id)
                gather_cond_x1_en = path_x1_en != -1
                mp_x1_en = torch.div(torch.bmm(gather_cond_x1_en.to(logits_x1_en).unsqueeze(1), logits_x1_en)[0], gather_cond_x1_en.sum(-1).unsqueeze(1)).squeeze()
                # get x1-x2 best alignment
                # get new random language tokens
                num_rand_langs = model.num_language - 2
                seq_langtokens = torch.randint(num_rand_langs, (ori_tgt_lang_id.size()), device=ori_tgt_lang_id.device, dtype=ori_tgt_lang_id.dtype) + model.langtok_offset 
                offset_cond = (seq_langtokens>=ori_tgt_lang_id)
                seq_langtokens[offset_cond] = seq_langtokens[offset_cond] + 1
                offset_cond = (seq_langtokens>=en_id)
                seq_langtokens[offset_cond] = seq_langtokens[offset_cond] + 1
                aug_tgt_lang_id = seq_langtokens
                aug_src_tokens = new_src_tokens.clone()
                aug_src_tokens[:, 0] = aug_tgt_lang_id
                links_x1_x2, logits_x1_x2 = model.dag_decode_with_pos(aug_src_tokens, new_src_lengths, new_prev_output_tokens, rand_seed=rand_seed, tgt_lang_id=aug_tgt_lang_id)
                gather_cond_x1_x2 = torch.zeros(path_x1_en.size(), device=logits_x1_en.device, dtype=logits_x1_en.dtype)
                for idx, link in enumerate(links_x1_x2):
                    for l in link:
                        gather_cond_x1_x2[idx][l] = 1.0
                mp_x1_x2 = torch.div(torch.bmm(gather_cond_x1_x2.unsqueeze(1), logits_x1_x2)[0], gather_cond_x1_x2.sum(-1).unsqueeze(1)).squeeze()
                _dec_consist_loss_loss_nonefactor = F.mse_loss(mp_x1_en, mp_x1_x2.detach())
                _dec_consist_loss_loss = _dec_consist_loss_loss_nonefactor * 0.1
                _dec_consist_loss = {"name": 'decoder_consist-loss', "loss": _dec_consist_loss_loss, "loss_nofactor": _dec_consist_loss_loss_nonefactor}
            else:
                _dec_consist_loss = {"name": 'decoder_consist-loss', "loss": torch.tensor(0.0).to(dag_nll_loss), "loss_nofactor": torch.tensor(0.0).to(dag_nll_loss)}
            losses += [_dec_consist_loss]
         

        # BT LOSS 
        bt_threshold = sample.get("bt_threshold", 100000)
        _bt_losses = None
        bt_dag_nll_loss = None
        if is_training and doing_bt and update_num > bt_threshold: # 
            pad_index = model.pad
            src_lang_id = sample['net_input']['src_lang_id'].clone()
            tgt_lang_id = sample['net_input']['tgt_lang_id'].clone()
            src_tokens = sample['net_input']['src_tokens'].clone()
            src_lengths = sample['net_input']['src_lengths'].clone()
            # target_tokens = sample['target']
            # target_lengths = target_tokens.ne(pad_index).sum(-1)

            bt_src_tokens = outputs.get("bt_output_tokens", 0)
            bt_src_tokens[:, 0] = 0
            bt_src_tokens_lengths = bt_src_tokens.ne(pad_index).sum(-1)
            
            LEN_FILTER_RATIO = 2
            filter_cond = torch.logical_and(bt_src_tokens_lengths * LEN_FILTER_RATIO > src_lengths, 
                                            src_lengths * LEN_FILTER_RATIO > bt_src_tokens_lengths)
            filter_cond = torch.logical_and(filter_cond, bt_src_tokens_lengths < 128)
            filter_cond = torch.logical_and(filter_cond, src_lengths < 128)
            if filter_cond.sum() > 0:
                if getattr(model.args, "csbt", False) or getattr(model.args, "seqbt", False):
                    bt_src_lang_id = outputs["bt_langtok"][filter_cond]
                else:
                    bt_src_lang_id = tgt_lang_id[filter_cond]
                bt_tgt_lang_id = src_lang_id[filter_cond]
                bt_src_tokens = bt_src_tokens[filter_cond]
                bt_src_tokens_lengths = bt_src_tokens_lengths[filter_cond]
                bt_target_tokens = src_tokens[filter_cond]
                bt_prev_output_tokens = model.initialize_output_tokens_by_tokens(bt_src_tokens, bt_target_tokens, None)

                bt_glat = glat if  getattr(model.args, 'bt_glat', False) else None
                bt_outputs = model(bt_src_tokens, bt_src_tokens_lengths, bt_prev_output_tokens, bt_target_tokens, bt_glat, glat_function, src_lang_id=bt_src_lang_id, tgt_lang_id=bt_tgt_lang_id)
                _bt_losses = self._compute_dag_loss(
                    bt_outputs["word_ins"].get("out"),
                    bt_prev_output_tokens.ne(self.task.target_dictionary.pad()),
                    bt_outputs["word_ins"].get("tgt"),
                    bt_outputs["word_ins"].get("mask", None),
                    bt_outputs["links"],
                    name="bt-dag-loss",
                    factor=0.5,
                    matchmask=bt_outputs.get('matchmask', None),
                    keep_word_mask=bt_outputs.get('keep_word_mask', None),
                    model=model,
                    ltc=bt_outputs["ltc"]
                )
            else:
                _bt_losses ={ "name": "bt-dag-loss",
                              "loss": torch.tensor(0.0).to(dag_nll_loss), 
                              "loss_nofactor": torch.tensor(0.0).to(dag_nll_loss),
                              "bt-dag_nll-loss": torch.tensor(0.0).to(dag_nll_loss)
                            }
        if _bt_losses is not None:
            losses += [_bt_losses]
            bt_dag_nll_loss = _losses.get("nll_loss", 0.0) 
        
        #length
        _losses = self._compute_loss(
            outputs["length"].get("out"),
            outputs["length"].get("tgt"),
            None,
            0,
            name="length-loss",
            factor=outputs["length"]["factor"], )
        losses += [_losses]
        length_nll_loss = _losses.get("nll_loss", 0.0)

        loss = sum(l["loss"] for l in losses)

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "dag_nll-loss": dag_nll_loss.data,
            "length_nll-loss": length_nll_loss.data,
            "ntokens": ntokens,
            "nvalidtokens": nvalidtokens,
            "nsentences": nsentences,
            "invalid_nsentences": invalid_nsentences,
            "sample_size": sample_size,
            "glat_acc": outputs.get("glat_accu", 0),
            "glat_keep": outputs.get("glat_keep", 0)
        }
        
        if bt_dag_nll_loss is not None:
            logging_output["bt-dag_nll-loss"] = bt_dag_nll_loss.data

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss_nofactor"])
                if reduce
                else l["loss_nofactor"]
            )
        # gpu_tracker.track()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )  # each batch is 1
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nvalidtokens = sum(log.get('nvalidtokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        invalid_nsentences = sum(log.get('invalid_nsentences', 0) for log in logging_outputs)
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))
       
        res = {
            "ntokens": utils.item(ntokens),
            "nsentences": utils.item(nsentences),
            "nvalidtokens": utils.item(nvalidtokens),
            "invalid_nsentences": utils.item(invalid_nsentences),
            'tokens_perc': utils.item(nvalidtokens / ntokens),
            'sentences_perc': 1 - utils.item(invalid_nsentences / nsentences),
        }
        res["loss"] = loss / sample_size
        res["glat_acc"] = glat_acc / sample_size
        res["glat_keep"] = glat_keep / sample_size
        
        for key, value in res.items():
            metrics.log_scalar(
                key, value, sample_size, round=3
            )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
