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
from lmnet.networks.super_resolution.{{network_module}} import {{network_class}}
from lmnet.datasets.{{dataset_module}} import {{dataset_class}}
{% if data_augmentation %}from lmnet.data_augmentor import ({% for augmentor in data_augmentation %}
    {{ augmentor[0] }},{% endfor %}
){% endif %}
from lmnet.data_processor import Sequence

IS_DEBUG = False

NETWORK_CLASS = {{network_class}}

DATASET_CLASS = type('DATASET_CLASS', ({{dataset_class}},), {{dataset_class_property}})

IMAGE_SIZE = {{image_size}}
BATCH_SIZE = {{batch_size}}
DATA_FORMAT = "NHWC"
TASK = Tasks.SUPER_RESOLUTION
CLASSES = {{classes}}

MAX_EPOCHS = {{max_epochs}}


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = None
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = {{optimizer_class}}
NETWORK.OPTIMIZER_KWARGS = {{optimizer_kwargs}}
NETWORK.LEARNING_RATE_FUNC = {{learning_rate_func}}
NETWORK.LEARNING_RATE_KWARGS = {{learning_rate_kwargs}}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([{% if data_augmentation %}{% for augmentor in data_augmentation %}
    {{ augmentor[0] }}({% for d_name, d_value in augmentor[1] %}{{ d_name }}={{ d_value }}, {% endfor %}),{% endfor %}
{% endif %}])
