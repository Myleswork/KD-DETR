import os

import math
import numpy as np
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, get_rank)

from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss)
from .transformer import build_transformer


class KDCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 loss_kd_logits=None,
                 aux_refpoints=None,
                 random_refpoints=None,
                 **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        loss_map = {'mse': self.loss_mse, 'kl':self.loss_kl_div}
        self.T = 10
        self.weight_dict = {}
        if loss_kd_logits is not None:
            self.loss_cls = self.loss_kl_div
            self.weight_dict.update({'loss_kd_cls':10})
            self.loss_bbox = self.loss_l1_smooth
            self.weight_dict.update({'loss_kd_bbox':1})
            self.with_match = loss_kd_logits

        if aux_refpoints:
            self.loss_kd_auxrf_cls = self.loss_kl_div
            self.loss_kd_auxrf_box = self.loss_boxes 
            self.weight_dict.update({'loss_kd_auxrf_cls':1})
            self.weight_dict.update({'loss_kd_auxrf_bbox':5})
            self.weight_dict.update({'loss_kd_auxrf_giou':2})

        if random_refpoints:
            self.loss_kd_randomrf_cls = self.loss_kl_div
            self.loss_kd_randomrf_box = self.loss_boxes 
            self.weight_dict.update({'loss_kd_randomrf_cls':1})
            self.weight_dict.update({'loss_kd_randomrf_bbox':5})
            self.weight_dict.update({'loss_kd_randomrf_giou':2})


    @property
    def with_aux_refpoints(self):
        return hasattr(self, 'loss_kd_auxrf_cls') and self.loss_kd_auxrf_cls is not None
    @property
    def with_random_refpoints(self):
        return hasattr(self, 'loss_kd_randomrf_cls') and self.loss_kd_randomrf_cls is not None



    @property
    def with_logits(self):
        return hasattr(self, 'loss_bbox') and self.loss_bbox is not None

    @property
    def with_adapt(self):
        return hasattr(self, 'adapt') and self.adapt is not None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_matched_outputs(self, indices, outputs):
        keys = ['pred_logits', 'pred_boxes']
        matched_outputs = dict(pred_logits=[],pred_boxes=[])
        for batch, indice in enumerate(indices):
           _, gt_index = torch.sort(indice[-1])
           output_index = indice[0][gt_index]
           #for k, v in outputs.items():
           for k in keys:
               matched_outputs[k].append(outputs[k][batch][output_index])
        for k in keys:
            outputs[k] = torch.vstack(matched_outputs[k])
        return outputs

    def forward(self,
        outputs,
        soft_targets,
        student_indices=None,
        teacher_indices=None,
        student_topk=None,
        teacher_topk=None):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        losses = {}

        if self.with_aux_refpoints:
            aux_weight = soft_targets['pred_logits'].flatten(0, 1)
            aux_weight = aux_weight.sigmoid().max(1)[0].detach()
 
            aux_outputs = outputs['auxrf']
            loss_kd_auxrf_cls = self.loss_kd_auxrf_cls(aux_outputs['pred_logits'],
                                                       soft_targets['pred_logits'], aux_weight)
            loss_tmp = self.loss_kd_auxrf_box(aux_outputs['pred_boxes'],
                                              soft_targets['pred_boxes'], aux_weight)
            loss_kd_auxrf_bbox = loss_tmp['loss_bbox']
            loss_kd_auxrf_giou = loss_tmp['loss_giou']

            """
            for lvl, (aux_output, aux_soft_output) in enumerate(zip(outputs['auxrf_aux_outputs'], soft_targets['aux_outputs'])):
                loss_kd_auxrf_cls += self.loss_kd_auxrf_cls(aux_outputs['pred_logits'], aux_soft_output['pred_logits'], weight)
                loss_tmp = self.loss_kd_auxrf_box(aux_outputs['pred_boxes'], aux_soft_output['pred_boxes'], weight)
                loss_kd_auxrf_bbox += loss_tmp['loss_bbox']
                loss_kd_auxrf_giou += loss_tmp['loss_giou']
            """
            for aux_output in aux_outputs['aux_outputs']:
                loss_kd_auxrf_cls += self.loss_kd_auxrf_cls(aux_output['pred_logits'],
                                                            soft_targets['pred_logits'], aux_weight)
                loss_tmp = self.loss_kd_auxrf_box(aux_output['pred_boxes'],
                                                  soft_targets['pred_boxes'], aux_weight)
                loss_kd_auxrf_bbox += loss_tmp['loss_bbox']
                loss_kd_auxrf_giou += loss_tmp['loss_giou']

            losses.update(dict(loss_kd_auxrf_cls=loss_kd_auxrf_cls,
                               loss_kd_auxrf_bbox=loss_kd_auxrf_bbox,
                               loss_kd_auxrf_giou=loss_kd_auxrf_giou))

        if self.with_random_refpoints:
            random_weight = soft_targets['randomrf']['pred_logits'].flatten(0, 1)
            random_weight = random_weight.sigmoid().max(1)[0].detach()
 
            aux_outputs = outputs['randomrf']
            aux_soft_targets = soft_targets['randomrf']
            loss_kd_randomrf_cls = self.loss_kd_randomrf_cls(aux_outputs['pred_logits'],
                                                       aux_soft_targets['pred_logits'], random_weight)
            loss_tmp = self.loss_kd_randomrf_box(aux_outputs['pred_boxes'],
                                              aux_soft_targets['pred_boxes'], random_weight)
            loss_kd_randomrf_bbox = loss_tmp['loss_bbox']
            loss_kd_randomrf_giou = loss_tmp['loss_giou']

            """
            for lvl, (aux_output, aux_soft_output) in enumerate(zip(outputs['auxrf_aux_outputs'], soft_targets['aux_outputs'])):
                loss_kd_auxrf_cls += self.loss_kd_auxrf_cls(aux_outputs['pred_logits'], aux_soft_output['pred_logits'], weight)
                loss_tmp = self.loss_kd_auxrf_box(aux_outputs['pred_boxes'], aux_soft_output['pred_boxes'], weight)
                loss_kd_auxrf_bbox += loss_tmp['loss_bbox']
                loss_kd_auxrf_giou += loss_tmp['loss_giou']
            """
            for aux_output in aux_outputs['aux_outputs']:
                loss_kd_randomrf_cls += self.loss_kd_randomrf_cls(aux_output['pred_logits'],
                                                            aux_soft_targets['pred_logits'], random_weight)
                loss_tmp = self.loss_kd_randomrf_box(aux_output['pred_boxes'],
                                                  aux_soft_targets['pred_boxes'], random_weight)
                loss_kd_randomrf_bbox += loss_tmp['loss_bbox']
                loss_kd_randomrf_giou += loss_tmp['loss_giou']

            losses.update(dict(loss_kd_randomrf_cls=loss_kd_randomrf_cls,
                               loss_kd_randomrf_bbox=loss_kd_randomrf_bbox,
                               loss_kd_randomrf_giou=loss_kd_randomrf_giou))


        if self.with_logits:
            if self.with_match:
                student_indice = student_indices[-1]
                outputs = self.get_matched_outputs(student_indice, outputs)
                teacher_indice = teacher_indices[-1]
                soft_targets = self.get_matched_outputs(teacher_indice, soft_targets)
              
            soft_cls = soft_targets['pred_logits']
            soft_bbox = soft_targets['pred_boxes']  

            loss_kd_cls = self.loss_cls(outputs['pred_logits'], soft_cls)
            loss_kd_bbox = self.loss_bbox(outputs['pred_boxes'], soft_bbox)

            for lvl, (aux_output, aux_soft_output) in enumerate(zip(outputs['aux_outputs'], soft_targets['aux_outputs'])):
                if self.with_match:
                    aux_output = self.get_matched_outputs(student_indices[lvl], aux_output)
                    aux_soft_output = self.get_matched_outputs(teacher_indices[lvl], aux_soft_output)  
                loss_kd_cls += self.loss_cls(aux_output['pred_logits'],
                        aux_soft_output['pred_logits'])
                loss_kd_bbox += self.loss_bbox(aux_output['pred_boxes'], 
                        aux_soft_output['pred_boxes'])
            losses.update({'loss_kd_cls':loss_kd_cls})
            losses.update({'loss_kd_bbox':loss_kd_bbox})

        return losses

    def loss_l1_smooth(self, pred, soft_label):
        assert pred.size() == soft_label.size()
        if len(pred.shape) == 3:
            pred = pred.flatten(0, 1)
            soft_label = soft_label.flatten(0, 1)
        target = soft_label.detach()
        loss = nn.SmoothL1Loss(beta=1.0/9.0)
        return loss(pred, target)

    def loss_l1(self, pred, soft_label, weight=None):
        assert pred.size() == soft_label.size()
        if len(pred.shape) == 3:
            pred = pred.flatten(0, 1)
            soft_label = soft_label.flatten(0, 1)
        target = soft_label.detach()
        loss = torch.abs(target - pred).mean(1)
        if weight is not None:
            loss = sum(loss * weight)
        else:
            loss = sum(loss)
        return loss
        

    def loss_kl_div(self, pred, soft_label, weight=None):
        assert pred.size() == soft_label.size()
       
        if len(pred.shape) > 2:
            pred = pred.flatten(start_dim=0, end_dim=-2)
            soft_label = soft_label.flatten(start_dim=0, end_dim=-2)
        
        target = F.softmax(soft_label / self.T, dim=1)
        target = target.detach()
        if weight is not None:
            kd_loss = F.kl_div(F.log_softmax(pred / self.T, dim=1), target, reduction='none').mean(1) * self.T * self.T 
            kd_loss = sum(kd_loss * weight)
        else:
            kd_loss = F.kl_div(F.log_softmax(pred / self.T, dim=1), target, reduction='none').mean(1) * (self.T * self.T)
            kd_loss = kd_loss.mean()

        return kd_loss

    def loss_mse(self, pred, soft_label):
        assert pred.size() == soft_label.size()
        import numpy as np
        pred[pred == -np.inf] = 0
        soft_label[soft_label == -np.inf] = 0
        soft_label = soft_label.detach()

        return F.mse_loss(pred, soft_label)

    def loss_mse_withlog(self, pred, soft_label):
        assert pred.size() == soft_label.size()
        if len(pred.shape) == 3:
            pred = pred.flatten(0, 1)
            soft_label = soft_label.flatten(0, 1)
        target = F.softmax(soft_label / self.T, dim=1)
        target = target.detach()
        kd_loss = F.mse_loss(F.softmax(pred / self.T, dim=1), target, reduction='sum') * self.T * self.T
        return kd_loss


    def loss_class(self, pred, soft_label):
        assert pred.size() == soft_label.size()
        pred = pred.flatten(0, 1)
        soft_label = soft_label.flatten(0, 1)
        weight = soft_label.sigmoid().max(1)[0]
        return self.loss_kl_div(pred, soft_label, weight)



    def loss_boxes(self, pred, soft_label, weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        losses = {}
        if len(pred.shape) == 3:
            pred = pred.flatten(0, 1)
            soft_label = soft_label.flatten(0, 1)
        loss_bbox = F.l1_loss(pred, soft_label, reduction='none').mean(1)
        losses['loss_bbox'] = sum(loss_bbox * weight) / weight.sum()

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred),
            box_ops.box_cxcywh_to_xyxy(soft_label)))
        losses['loss_giou'] = sum(loss_giou * weight) / weight.sum()

        return losses


