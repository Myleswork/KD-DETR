# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from .DAB_DETR import build_DABDETR, build_kd_DABDETR, KDCriterion

try:
    from .dab_deformable_detr import build_dab_deformable_detr, build_kd_dab_deformable_detr
except:
    pass
