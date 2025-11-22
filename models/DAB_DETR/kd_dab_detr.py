# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------




import os

import math
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .kd_matcher import build_kd_matcher
from .transformer import build_transformer
from .kd_transformer import build_kd_transformer
from .kd_loss import KDCriterion
from .kd_setcriterion import KDSetCriterion

from .DABDETR import (sigmoid_focal_loss, DABDETR,
                      PostProcess, build_DABDETR, MLP)


class KD_DABDETR(DABDETR):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=False,
                    random_refpoints_xy=False,
                    aux_refpoints=None,
                    return_weightmap=False,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
            

        """
        super(KD_DABDETR, self).__init__(backbone, transformer, num_classes, num_queries,
                                         aux_loss=aux_loss,
                                         iter_update=iter_update,
                                         query_dim=query_dim,
                                         bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                         random_refpoints_xy=random_refpoints_xy)
        
        self.aux_refpoints = aux_refpoints
        self.return_weightmap = return_weightmap

    @property
    def with_aux_refpoints(self):
        return hasattr(self, 'aux_refpoints') and self.aux_refpoints is not None

    def freeze_decoder(self):
        self.transformer.decoder.freeze_params()
        for p in self.bbox_embed.parameters():
            p.requires_grad = False
        for p in self.class_embed.parameters():
            p.requires_grad = False
        self.refpoint_embed.requires_grad = False

    def forward(self, samples: NestedTensor, aux_refpoints=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1][0].decompose()
        src_wo_relu = features[-1][1]
        assert mask is not None
        # default pipeline
        embedweight = self.refpoint_embed.weight

        auxrfp = self.aux_refpoints.detach() if self.with_aux_refpoints else None
        randomrfp = aux_refpoints
        
        hs, reference, weightmaps, hs_aux, reference_aux, weightmaps_aux, \
        hs_random, reference_random, weightmaps_random, memory = self.transformer(self.input_proj(src), mask, embedweight,
                                                                 pos[-1], auxrfp, randomrfp)
        out = self.get_outputs(hs, reference)
        out['weight_maps'] = weightmaps
        if hs_aux is not None:
            aux_out = self.get_outputs(hs_aux, reference_aux)
            out['auxrf'] = aux_out
            out['auxrf_weight_maps'] = weightmaps_aux
            
        if hs_random is not None:
            random_out = self.get_outputs(hs_random, reference_random)
            out['randomrf'] = random_out
            out['randomrf_weight_maps'] = weightmaps_random

        return out
 

    def get_outputs(self, hs, reference):
        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out


def build_kd_DABDETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    device = torch.device(args.device)

    teacher_backbone = build_backbone(args, teacher=True)
    return_weightmap = False
    teacher_transformer = build_kd_transformer(args, return_weightmap=return_weightmap, teacher=True)
    teacher_model = KD_DABDETR(
        teacher_backbone,
        teacher_transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        return_weightmap = return_weightmap
    )
    teacher_ckpt = torch.load('./pretrained/dab-detr-r50.pth', map_location='cpu')
    teacher_model.load_state_dict(teacher_ckpt['model'])

    aux_refpoints = teacher_model.refpoint_embed.weight if args.aux_refpoints else None

    student_backbone = build_backbone(args)
    student_transformer = build_kd_transformer(args, return_weightmap=return_weightmap)
    student_model = KD_DABDETR(
        student_backbone,
        student_transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        aux_refpoints = aux_refpoints,
        return_weightmap = return_weightmap
    )

    matcher = build_kd_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = KDSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)

    config = vars(args)
    kd_criterion = KDCriterion(512, 2048, **config)
    kd_criterion.to(device)

    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}

    return student_model, teacher_model, criterion, kd_criterion, postprocessors


