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
from lmnet.networks.super_resolution.dcscn import Dcscn
from lmnet.datasets.div2k import Div2kSuperResolution

from lmnet.data_processor import Sequence

from lmnet.pre_processor import Scale
from lmnet.post_processor import ConvertYAndCbcrToRgb
from lmnet.data_augmentor import Crop, RgbToYcbcr

IS_DEBUG = False

NETWORK_CLASS = Dcscn
DATASET_CLASS = Div2kSuperResolution

SCALE = 2
PATCH_SIZE = 96

IMAGE_SIZE = [128, 128]
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
# NETWORK.OPTIMIZER_KWARGS = {'learning_rate': 0.00005, 'beta1': 0.9, 'beta2': 0.999}
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

NETWORK.INPUT_CHANNEL = 1
NETWORK.OUTPUT_CHANNEL = 1

# Scale factor for Super Resolution (should be 2 or more)
NETWORK.SCALE = SCALE
# Number of feature extraction layers
NETWORK.FEATURE_EXTRACTION_LAYERS = 12
# Number of filters of first feature-extraction CNNs
NETWORK.FIRST_FEATURE_EXTRACTION_LAYER_FILTERS = 196
# Number of filters of last feature-extraction CNNs
NETWORK.LAST_FEATURE_EXTRACTION_LAYER_FILTERS = 48
# Number of CNN filters are decayed from [first] to [last] by this gamma
NETWORK.FILTERS_DECAY_GAMMA = 1.5

DATASET = EasyDict()
DATASET.SCALE = SCALE
DATASET.PATCH_SIZE = PATCH_SIZE
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = None
