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

IS_DEBUG = False

NETWORK_CLASS = Dcscn
DATASET_CLASS = Div2kSuperResolution

SCALE = 2
CROP_SIZE = 48

IMAGE_SIZE = [128, 128]
BATCH_SIZE = 10
DATA_FORMAT = "NHWC"
TASK = Tasks.SUPER_RESOLUTION
CLASSES = []

MAX_EPOCHS = 1
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
    Scale(1/SCALE),
])
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {'momentum': 0.9, 'learning_rate': 0.001}
NETWORK.LEARNING_RATE_FUNC = None
NETWORK.LEARNING_RATE_KWARGS = None

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

DATASET = EasyDict()
DATASET.SCALE = SCALE
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([Crop(size=CROP_SIZE)])
