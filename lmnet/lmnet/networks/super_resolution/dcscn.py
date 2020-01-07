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

import numpy as np
import tensorflow as tf

from lmnet.networks.base import BaseNetwork
from lmnet.layers import conv2d
from lmnet.blocks import conv_bn_act
from lmnet.utils.image import convert_ycbcr_to_rgb, convert_y_and_cbcr_to_rgb, convert_rgb_to_ycbcr


class Dcscn(BaseNetwork):
    """ """

    def __init__(
        self,
        scale=2,
        input_channel=1,
        output_channel=1,
        feature_extraction_layers=None,
        weight_decay_rate=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_channel = output_channel
        self.scale = scale
        self.filters = [] if feature_extraction_layers is None else feature_extraction_layers
        self.weight_decay_rate = weight_decay_rate
        self.activation = tf.nn.leaky_relu

        # Output nodes should be kept by this probability. If 1, don't use dropout.
        self.dropout_rate = 0.8

        # Use batch normalization after each CNNs
        self.batch_norm = False

        self.custom_getter = None

    def placeholders(self):
        label_height = self.image_size[0] * self.scale if self.image_size[0] is not None else None
        label_width = self.image_size[1] * self.scale if self.image_size[1] is not None else None

        batch_size = self.batch_size if self.image_size[0] is not None else None

        x = tf.placeholder(tf.float32, shape=[batch_size, self.image_size[0], self.image_size[1], 3], name="x")
        y = tf.placeholder(tf.float32, shape=[batch_size, label_height, label_width, 3], name="y")

        return x, y
    
    def feature_extraction_base(self, input, is_training):
        prev_output = input

        # Feature extraction layer
        for i, filter_num in enumerate(self.filters):
            output = conv_bn_act(
                "conv{}".format(i + 1),
                inputs=prev_output,
                filters=filter_num,
                kernel_size=3,
                weight_decay_rate=self.weight_decay_rate,
                activation=self.activation,
                is_training=is_training
            )

            with tf.variable_scope("conv{}".format(i + 1)):
                if i == 0:
                    pass
                elif i == 1:
                    concated = tf.concat([prev_output, output], 3)
                else:
                    concated = tf.concat([concated, output], 3)

            prev_output = output
        
        return concated
    
    def reconstruction_base(self, input, is_training):
        # Reconstruction layer
        a_filters = 64
        b_filters = 32

        a1_output = conv_bn_act(
            "a1",
            inputs=input,
            filters=a_filters,
            kernel_size=1,
            weight_decay_rate=self.weight_decay_rate,
            activation=self.activation,
            is_training=is_training
        )
        b1_output = conv_bn_act(
            "b1",
            inputs=input,
            filters=b_filters,
            kernel_size=1,
            weight_decay_rate=self.weight_decay_rate,
            activation=self.activation,
            is_training=is_training
        )
        b2_output = conv_bn_act(
            "b2",
            inputs=b1_output,
            filters=b_filters,
            kernel_size=3,
            weight_decay_rate=self.weight_decay_rate,
            activation=self.activation,
            is_training=is_training
        )
        recon_output = tf.concat([b2_output, a1_output], 3, name="Concat2")

        # Upsampling layer
        upsample_output = conv_bn_act(
            "up-ps",
            inputs=recon_output,
            filters=self.scale*self.scale*(a_filters+b_filters),
            kernel_size=3,
            weight_decay_rate=self.weight_decay_rate,
            activation=self.activation,
            is_training=is_training
        )
        upsample_output = tf.depth_to_space(upsample_output, self.scale)

        with tf.compat.v1.variable_scope("r_conv"):
            network_output = tf.layers.conv2d(
                inputs=upsample_output,
                filters=self.output_channel,
                kernel_size=3,
                padding="SAME",
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation=None,
                use_bias=False
            )

        return network_output

    def base(self, x, is_training):
        # tf.summary.image("input", x)
        shape_of_x = tf.shape(x)
        height = shape_of_x[1]
        width = shape_of_x[2]

        x2 = tf.image.resize_images(
            x,
            (height * 2, width * 2),
            method=tf.image.ResizeMethod.BICUBIC
        )

        feature_extraction_output = self.feature_extraction_base(x, is_training)
        reconstruction_output = self.reconstruction_base(feature_extraction_output, is_training)

        y_hat = tf.add(reconstruction_output, x2, name="output")

        return y_hat

    def inference(self, x_placeholder, is_training):
        y_image = x_placeholder[:, :, :, 0:1]
        y_hat = self.base(y_image, is_training=is_training)

        output = tf.identity(y_hat, name="output")

        return output

    def loss(self, output, y_placeholder):
        with tf.name_scope("loss"):
            y_image = y_placeholder[:, :, :, 0:1]
            diff = tf.subtract(output, y_image, "diff")

            mse = tf.reduce_mean(tf.square(diff, name="diff_square"), name="mse")
            loss = tf.identity(mse, name="image_loss")

            weight_decay_loss = tf.losses.get_regularization_loss()
            loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)
            tf.summary.scalar("weight_decay", weight_decay_loss)

            return loss
    
    def _ycbcr_to_rgb(self, images):
        rgb_images = []
        for image in images:
            image = convert_ycbcr_to_rgb(image)
            # image.round().clip(0, 255)
            rgb_images.append(image)
        return np.array(np.round(rgb_images), dtype=np.float32)
        
    def _combine_y_and_cbcr_to_rgb(self, y_images, ycbcr_images):
        combined_images = []
        for y_image, ycbcr_image in zip(y_images, ycbcr_images):
            image = convert_y_and_cbcr_to_rgb(y_image, ycbcr_image[:, :, 1:3])
            combined_images.append(image)
        
        return np.array(combined_images, dtype=np.float32)

    def summary(self, output, labels):
        """[summary]
        
        Args:
            output ([type]): [description]
            labels (np.array): 4-D numpy array of YCbCr image
        
        Returns:
            [type]: [description]
        """
        shape_of_bicubic_image = tf.shape(labels)
        height = shape_of_bicubic_image[1]
        width = shape_of_bicubic_image[2]

        bicubic = tf.image.resize_images(labels, (height // 2, width // 2), tf.image.ResizeMethod.BICUBIC)
        bicubic = tf.image.resize_images(bicubic, (height, width), tf.image.ResizeMethod.BICUBIC)

        tf.summary.image("output_image_Y", tf.cast(tf.clip_by_value(output, 0, 255), tf.uint8))
        tf.summary.image("grand_truth_Y", tf.cast(tf.clip_by_value(labels[:, :, :, 0:1], 0, 255), tf.uint8))
        tf.summary.image("bicubic_Y", tf.cast(tf.clip_by_value(bicubic[:, :, :, 0:1], 0, 255), tf.uint8))

        tf.summary.image("output_image", tf.cast(tf.clip_by_value(tf.py_func(self._combine_y_and_cbcr_to_rgb, [output, bicubic], tf.float32), 0, 255), tf.uint8))
        tf.summary.image("grand_truth", tf.cast(tf.clip_by_value(tf.py_func(self._ycbcr_to_rgb, [labels], tf.float32), 0, 255), tf.uint8))
        tf.summary.image("bicubic", tf.cast(tf.clip_by_value(tf.py_func(self._ycbcr_to_rgb, [bicubic], tf.float32), 0, 255), tf.uint8))

        return super().summary(output, labels)
    
    def metrics(self, output, labels):
        # Make RGB grand truth image
        gt_images = tf.py_func(self._ycbcr_to_rgb, [labels], tf.float32)

        # Make RGB image from output
        size = tf.shape(labels)
        height = size[1]
        width = size[2]

        base = tf.image.resize_images(labels, (height // 2, width // 2), tf.image.ResizeMethod.BICUBIC)
        base = tf.image.resize_images(base, (height, width), tf.image.ResizeMethod.BICUBIC)
        rgb_output = tf.py_func(self._combine_y_and_cbcr_to_rgb, [output, base], tf.float32)

        gt_images = tf.clip_by_value(gt_images, 0, 255)
        rgb_output = tf.clip_by_value(rgb_output, 0, 255)
 
        # Calc metrics
        results = {}
        updates = []
        with tf.name_scope('metrics_cals'):
            mean_squared_error, mean_squared_error_update = tf.metrics.mean_squared_error(
                gt_images,
                rgb_output,
            )
            results["mean_squared_error"] = mean_squared_error
            updates.append(mean_squared_error_update)

            psnr_array = tf.image.psnr(tf.cast(gt_images, tf.uint8), tf.cast(rgb_output, tf.uint8), max_val=255)
            psnr, psnr_update = tf.metrics.mean(psnr_array)
            results["psnr"] = psnr
            updates.append(psnr_update)

            ssim_array = tf.image.ssim(tf.cast(gt_images, tf.uint8), tf.cast(rgb_output, tf.uint8), max_val=255)
            ssim, ssim_update = tf.metrics.mean(ssim_array)
            results["ssim"] = ssim
            updates.append(ssim_update)

            # merge all updates
            updates_op = tf.group(*updates)

            return results, updates_op


class DcscnQuantize(Dcscn):
    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        # Quantizers
        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(
            self._quantized_variable_getter,
            weight_quantization=weight_quantization
        )

        # Embedding
        embedding_granularity = 256
        embedding_dim = 10
        embedding_initial_value = np.random.rand(embedding_granularity, embedding_dim).astype(np.float32)
        self.embedding = tf.Variable(embedding_initial_value)

    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)

    def encode_image(self, images):
        # images = tf.cast(images * 255, tf.int32)
        # images = tf.clip_by_value(images, 0, 255)
        encoded = tf.contrib.layers.embedding_lookup_unique(self.embedding, images)
        shape = encoded.get_shape()
        return tf.reshape(encoded, (shape[0], shape[1], shape[2], -1))

    def feature_extraction_base(self, input, is_training):
        input = self.encode_image(input)
        return super().feature_extraction_base(input, is_training)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.compat.v1.variable_scope(name):
            print(var.op.name)
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var

        # assert callable(weight_quantization)
        # var = getter(name, *args, **kwargs)
        # with tf.variable_scope(name):
        #     # Apply weight quantize to variable whose last word of name is "kernel".
        #     if "kernel" == var.op.name.split("/")[-1]:

        #         if "float" in var.op.name:
        #             print("not quantized", var.op.name)
        #             return var

        #         print("quantized", var.op.name)
        #         return weight_quantization(var)
        # return var
