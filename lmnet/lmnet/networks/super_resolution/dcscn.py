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
from lmnet.utils.image import convert_ycbcr_to_rgb, convert_y_and_cbcr_to_rgb, scale


class Dcscn(BaseNetwork):
    """ """

    def __init__(
        self,
        scale=2,
        input_channel=1,
        output_channel=1,
        feature_extraction_layers=12,
        first_feature_extraction_layer_filters=196,
        last_feature_extraction_layer_filters=48,
        filters_decay_gamma=1.5,
        weight_decay_rate=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_channel = output_channel
        self.scale = scale
        self.layers = feature_extraction_layers
        self.filters = first_feature_extraction_layer_filters
        self.min_filters = last_feature_extraction_layer_filters
        self.filters_decay_gamma = filters_decay_gamma
        self.weight_decay_rate = weight_decay_rate

        # Output nodes should be kept by this probability. If 1, don't use dropout.
        self.dropout_rate = 0.8

        # Use batch normalization after each CNNs
        self.batch_norm = False

        self.custom_getter = None

    def _conv2d(self, input, w, stride, bias=None, use_batch_norm=False, name=""):
        output = tf.nn.conv2d(
            input,
            w,
            strides=[1, stride, stride, 1],
            padding="SAME",
            name=name + "_conv",
        )

        if bias is not None:
            output = tf.add(output, bias, name=name + "_add")

        if use_batch_norm:
            output = tf.layers.batch_normalization(
                output, training=self.is_training, name="BN"
            )

        return output

    def _prelu(self, input, features, name=""):
        with tf.variable_scope("prelu"):
            alphas = tf.Variable(
                tf.constant(0.1, shape=[features]), name=name + "_prelu"
            )

        output = tf.nn.relu(input) + tf.multiply(alphas, (input - tf.abs(input))) * 0.5
        return output

    def _convolutional_block(
        self,
        name,
        input,
        kernel_size,
        filters,
        is_training,
        use_batch_norm=False,
        dropout_rate=1.0,
    ):
        with tf.variable_scope(name, custom_getter=self.custom_getter):
            a = tf.layers.conv2d(
                inputs=input,
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding="SAME",
                # TODO: prelu
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay_rate),
            )
            
            if dropout_rate < 1.0:
                a = tf.nn.dropout(a, dropout_rate, name="dropout")

            # Save image
            # shapes = w.get_shape().as_list()
            # weights = tf.reshape(w, [shapes[0], shapes[1], shapes[2] * shapes[3]])
            # weights_transposed = tf.transpose(weights, [2, 0, 1])
            # weights_transposed = tf.reshape(
            #     weights_transposed, [shapes[2] * shapes[3], shapes[0], shapes[1], 1]
            # )
            # tf.summary.image("weights", weights_transposed, max_outputs=6)

        return a

    def _pixel_shuffler(
        self, name, input, kernel_size, scale, filters, is_training
    ):
        with tf.variable_scope(name):
            output = self._convolutional_block(
                name + "_CNN",
                input,
                kernel_size,
                filters=scale * scale * filters,
                use_batch_norm=False,
                is_training=is_training
            )

            return output

    def placeholders(self):
        x = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="x"
        )
        y = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="y"
        )
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        return x, y

    def get_filters(self, first, last, layers, decay):
        return [
            int((first - last) * (1 - pow(i / float(layers - 1), 1.0 / decay)) + last)
            for i in range(layers)
        ]
    
    def feature_extraction_base(self, input, is_training):
        filters = self.get_filters(self.filters, self.min_filters, self.layers, self.filters_decay_gamma)
        outputs = []

        # Feature extraction layer
        for i, filter_num in enumerate(filters):
            output = self._convolutional_block(
                "CNN{}".format(i + 1),
                input,
                kernel_size=3,
                filters=filter_num,
                use_batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                is_training=is_training
            )
            outputs.append(output)
            input = output
        
        with tf.variable_scope("Concat"):
            network_output = tf.concat(outputs, 3, name="H_concat")
        
        return network_output
    
    def reconstruction_base(self, input, is_training):
        # Reconstruction layer
        a_filters = 64
        b_filters = 32

        a1_output = self._convolutional_block(
            "A1",
            input,
            kernel_size=1,
            filters=a_filters,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        b1_output = self._convolutional_block(
            "B1",
            input,
            kernel_size=1,
            filters=b_filters,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        b2_output = self._convolutional_block(
            "B2",
            b1_output,
            kernel_size=3,
            filters=b_filters,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        recon_output = tf.concat([b2_output, a1_output], 3, name="Concat2")

        # Upsampling layer
        upsample_output = self._convolutional_block(
            "Up-PS",
            recon_output,
            kernel_size=3,
            filters=self.scale*self.scale*(a_filters+b_filters),
            is_training=is_training
        )
        upsample_output = tf.depth_to_space(upsample_output, self.scale)

        network_output = self._convolutional_block(
            "R-CNN0",
            upsample_output,
            kernel_size=3,
            filters=self.output_channel,
            is_training=is_training
        )

        return network_output


    def base(self, x, is_training):
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
        y_image = tf.slice(x_placeholder, [0, 0, 0, 0], [1, -1, -1, 1])
        y_hat = self.base(y_image, is_training=is_training)

        output = tf.identity(y_hat, name="output")

        return output

    def loss(self, output, y_placeholder):
        with tf.name_scope("loss"):
            y_image = tf.slice(y_placeholder, [0, 0, 0, 0], [1, -1, -1, 1])
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
            rgb_images.append(convert_ycbcr_to_rgb(image))
        return np.array(np.round(rgb_images), dtype=np.uint8)
    
    def _rgb_output(self, output, labels):
        rgb_images = []
        for image, label in zip(output, labels):
            base_image = scale(scale(label, 1 / 2), 2)
            rgb_image = convert_y_and_cbcr_to_rgb(image, base_image[:, :, 1:3])
            rgb_image = rgb_image.round().clip(0, 200).astype(np.uint8)
            rgb_images.append(rgb_image)
        
        return np.array(rgb_images, dtype=np.uint8)
    
    def _bicubic(self, images):
        resized_images = []
        for image in images:
            image = scale(scale(image, 1 / 2), 2)
            resized_images.append(image)
        
        return np.array(resized_images, dtype=np.uint8)

    def summary(self, output, labels):
        tf.summary.image("output_image_Y", output)
        tf.summary.image("grand_truth_Y", tf.slice(labels, [0, 0, 0, 0], [1, -1, -1, 1]))
        tf.summary.image("bicubic_Y", tf.slice(
            tf.py_func(self._bicubic, [labels], tf.uint8), [0, 0, 0, 0], [1, -1, -1, 1]
        ))

        label_rgb = tf.py_func(self._ycbcr_to_rgb, [labels], tf.uint8)
        tf.summary.image("output_image", tf.py_func(self._rgb_output, [output, labels], tf.uint8))
        tf.summary.image("grand_truth", label_rgb)
        tf.summary.image("bicubic", tf.py_func(self._bicubic, [label_rgb], tf.uint8))

        return super().summary(output, labels)
        
    def _combine_y_and_cbcr_to_rgb(self, y_images, ycbcr_images):
        if y_images.ndim != ycbcr_images.ndim:
            raise Exception("The dimension of y_images and ycbcr_images must be the same.")
        
        combined_images = []
        for y_image, ycbcr_image in zip(y_images, ycbcr_images):
            image = convert_y_and_cbcr_to_rgb(y_image, ycbcr_image[:, :, 1:3])
            # TODO: この行いらないっぽい？
            # image = image.round().clip(0, 200).astype(np.float32)
            combined_images.append(image)
        
        return np.array(combined_images, dtype=np.float32)
    
    def metrics(self, output, labels):
        # Make grand truth image
        rgb_labels = tf.py_func(convert_ycbcr_to_rgb, [labels], tf.float32)

        # Make RGB image from output
        size = tf.shape(labels)
        height = size[1]
        width = size[2]

        base_images = tf.image.resize_images(
            labels,
            (height // 2, width // 2),
            method=tf.image.ResizeMethod.BICUBIC
        )
        base_images = tf.image.resize_images(
            base_images,
            (height, width),
            method=tf.image.ResizeMethod.BICUBIC
        )
        rgb_output = tf.py_func(self._combine_y_and_cbcr_to_rgb, [output, base_images], tf.float32)
 
        results = {}
        updates = []
        with tf.name_scope('metrics_cals'):
            mean_squared_error, mean_squared_error_update = tf.metrics.mean_squared_error(
                rgb_labels,
                rgb_output,
            )
            results["mean_squared_error"] = mean_squared_error
            updates.append(mean_squared_error_update)

            psnr_array = tf.image.psnr(rgb_labels, rgb_output, max_val=255)
            psnr, psnr_update = tf.metrics.mean(psnr_array)
            results["psnr"] = psnr
            updates.append(psnr_update)

            ssim_array = tf.image.ssim(rgb_labels, rgb_output, max_val=255)
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

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.compat.v1.variable_scope(name):
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
