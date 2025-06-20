# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from src.baselines.external.dinov2.dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from src.baselines.external.dinov2.dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg
from src.baselines.external.dinov2.dinov2.hub.classifiers import dinov2_vitb14_lc, dinov2_vitg14_lc, dinov2_vitl14_lc, dinov2_vits14_lc
from src.baselines.external.dinov2.dinov2.hub.classifiers import dinov2_vitb14_reg_lc, dinov2_vitg14_reg_lc, dinov2_vitl14_reg_lc, dinov2_vits14_reg_lc
from src.baselines.external.dinov2.dinov2.hub.depthers import dinov2_vitb14_ld, dinov2_vitg14_ld, dinov2_vitl14_ld, dinov2_vits14_ld
from src.baselines.external.dinov2.dinov2.hub.depthers import dinov2_vitb14_dd, dinov2_vitg14_dd, dinov2_vitl14_dd, dinov2_vits14_dd


dependencies = ["torch"]
