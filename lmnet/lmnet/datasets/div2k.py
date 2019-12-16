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
import functools
import os
from glob import glob

from lmnet.datasets.base import Base
from lmnet.datasets.base import SuperResolutionBase
from lmnet.utils.image import load_image


class Div2k(Base):
    classes = []
    num_classes = 0
    extend_dir = "DIV2K"
    available_subsets = ["train", "validation"]

    @property
    @functools.lru_cache(maxsize=None)
    def files(self):
        if self.subset == "train":
            images_dir = os.path.join(self.data_dir, "DIV2K_train_HR")
        else:
            images_dir = os.path.join(self.data_dir, "DIV2K_valid_HR")

        return [filepath for filepath in glob(os.path.join(images_dir, "*.png"))]

    @property
    def num_per_epoch(self):
        return len(self.files)

    def __getitem__(self, i, type=None):
        target_file = self.files[i]
        image = load_image(target_file)

        return image, None

    def __len__(self):
        return self.num_per_epoch


class Div2kSuperResolution(SuperResolutionBase):
    classes = Div2k.classes
    num_classes = Div2k.num_classes
    extend_dir = Div2k.extend_dir
    available_subsets = Div2k.available_subsets

    def __init__(self, **kwargs):
        self.dataset = Div2k(**kwargs)

        self.subset = self.dataset.subset
        self.batch_size = self.dataset.batch_size
        self.augmentor = self.dataset.augmentor
        self.pre_processor = self.dataset.pre_processor
        self.data_format = self.dataset.data_format
        self.seed = self.dataset.seed
        self.augment_subset = ["train", "validation"]

    @property
    def files(self):
        return self.dataset.files
    
    @property
    def num_per_epoch(self):
        return self.dataset.num_per_epoch
    
    def __getitem__(self, i):
        image, _ = self.dataset[i]
        return image, image
    
    def __len__(self):
        return len(self.dataset)
