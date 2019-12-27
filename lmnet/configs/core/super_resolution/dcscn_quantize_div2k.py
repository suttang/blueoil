# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.super_resolution.dcscn import DcscnQuantize
from lmnet.datasets.div2k import Div2kSuperResolution

from lmnet.data_processor import Sequence

from lmnet.pre_processor import Scale
from lmnet.post_processor import ConvertYAndCbcrToRgb
from lmnet.data_augmentor import Crop, RgbToYcbcr
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer
)

IS_DEBUG = False

NETWORK_CLASS = DcscnQuantize
DATASET_CLASS = Div2kSuperResolution

SCALE = 2
IMAGE_SIZE = [224, 224]
BATCH_SIZE = 1
DATA_FORMAT = "NHWC"
TASK = Tasks.SUPER_RESOLUTION
CLASSES = []

MAX_EPOCHS = 80
SAVE_CHECKPOINT_STEPS = 1000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 100


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# PRE_PROCESSOR = None
PRE_PROCESSOR = Sequence([
    RgbToYcbcr(with_keys=('image', 'mask')),
])
POST_PROCESSOR = Sequence([
    ConvertYAndCbcrToRgb(scale=SCALE)
])

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
# NETWORK.OPTIMIZER_KWARGS = {'learning_rate': 0.0001, 'beta1': 0.9, 'beta2': 0.999}
# NETWORK.LEARNING_RATE_FUNC = None
# NETWORK.LEARNING_RATE_KWARGS = None
NETWORK.OPTIMIZER_KWARGS = {'beta1': 0.9, 'beta2': 0.999}
NETWORK.LEARNING_RATE_FUNC = tf.train.cosine_decay
NETWORK.LEARNING_RATE_KWARGS = {
    "learning_rate": 0.0003,
    "decay_steps": 80 * 800,
}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0001

NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 1,
    'max_value': 1
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

NETWORK.INPUT_CHANNEL = 1
NETWORK.OUTPUT_CHANNEL = 1

# Scale factor for Super Resolution (should be 2 or more)
NETWORK.SCALE = SCALE
# Number of feature extraction layers
# NETWORK.FEATURE_EXTRACTION_LAYERS = [196, 166, 148, 133, 120, 108, 97, 86, 76, 66, 57, 48]
NETWORK.FEATURE_EXTRACTION_LAYERS = [192, 160, 160, 128, 128, 96, 96, 96, 64, 64, 64, 32]

DATASET = EasyDict()
DATASET.SCALE = SCALE
DATASET.PATCH_SIZE = IMAGE_SIZE
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = None
