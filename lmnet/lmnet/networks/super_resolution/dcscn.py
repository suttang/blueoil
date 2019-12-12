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
import math

import tensorflow as tf

from lmnet.networks.base import BaseNetwork
from lmnet.layers import conv2d


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
        self.input_channel = input_channel
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

        self.Weights = []
        self.Biases = []

        self.mse = None
        self.image_loss = None

    def _weight(self, shape, name="weight"):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        return tf.Variable(initializer(shape), name=name)

    def _bias(self, shape, initial_value=0.0, name="bias"):
        initial = tf.constant(initial_value, shape=shape)
        return tf.Variable(initial, name=name)

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
        output_feature_num,
        is_training,
        use_batch_norm=False,
        dropout_rate=1.0,
    ):
        with tf.variable_scope(name):
            a = tf.layers.conv2d(
                inputs=input,
                filters=output_feature_num,
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
        self, name, input, kernel_size, scale, output_feature_num, is_training
    ):
        with tf.variable_scope(name):
            output = self._convolutional_block(
                name + "_CNN",
                input,
                kernel_size,
                output_feature_num=scale * scale * output_feature_num,
                use_batch_norm=False,
                is_training=is_training
            )

            return output

    def placeholders(self):
        x = tf.placeholder(
            tf.float32, shape=[None, None, None, self.input_channel], name="x"
        )
        y = tf.placeholder(
            tf.float32, shape=[None, None, None, self.output_channel], name="y"
        )
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        return x, y

    def get_filters(self, first, last, layers, decay):
        return [
            int((first - last) * (1 - pow(i / float(layers - 1), 1.0 / decay)) + last)
            for i in range(layers)
        ]
        

    def base(self, x, is_training):
        tf.summary.image("input_image", x)

        # building feature extraction layers
        output_feature_num = self.filters
        total_output_feature_num = 0
        input_tensor = x
        
        input_shape = tf.shape(x)
        height = input_shape[1]
        width = input_shape[2]

        x2 = tf.image.resize_images(
            input_tensor,
            (height * 2, width * 2),
            method=tf.image.ResizeMethod.BICUBIC
        )

        tf.summary.image("input_bicubic_image", x2)

        feature_extraction_outputs = []
        for i, filters in enumerate(self.get_filters(self.filters, self.min_filters, self.layers, self.filters_decay_gamma)):
            output = self._convolutional_block(
                "CNN{}".format(i + 1),
                input_tensor,
                kernel_size=3,
                output_feature_num=filters,
                use_batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                is_training=is_training
            )
            feature_extraction_outputs.append(output)
            input_tensor = output
            total_output_feature_num += filters
        
        with tf.variable_scope("Concat"):
            feature_extraction_output = tf.concat(feature_extraction_outputs, 3, name="H_concat")

        # building reconstruction layers
        recon_a1_output = self._convolutional_block(
            "A1",
            feature_extraction_output,
            kernel_size=1,
            output_feature_num=64,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        recon_b1_output = self._convolutional_block(
            "B1",
            feature_extraction_output,
            kernel_size=1,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        recon_b2_output = self._convolutional_block(
            "B2",
            recon_b1_output,
            kernel_size=3,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        recon_output = tf.concat([recon_b2_output, recon_a1_output], 3, name="Concat2")

        # building upsampling layer
        pixel_shuffler_channel = 64 + 32
        upsample_output = self._pixel_shuffler(
            "Up-PS",
            recon_output,
            kernel_size=3,
            scale=self.scale,
            output_feature_num=pixel_shuffler_channel,
            is_training=is_training
        )
        upsample_output = tf.depth_to_space(upsample_output, self.scale)

        output = self._convolutional_block(
            "R-CNN0",
            upsample_output,
            kernel_size=3,
            output_feature_num=self.output_channel,
            is_training=is_training
        )

        y_hat = tf.add(output, x2, name="output")

        # with tf.name_scope("Y_"):
        #     mean = tf.reduce_mean(y_hat)
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(y_hat - mean)))
        #     tf.summary.scalar("output/mean", mean)
        #     tf.summary.scalar("output/stddev", stddev)
        #     tf.summary.histogram("output", y_hat)

        # tf.summary.image("output_image", y_hat)

        return y_hat

    def inference(self, x_placeholder, is_training):
        y_hat = self.base(x_placeholder, is_training=is_training)
        # output = self.post_process(x_placeholder, y_hat)

        self.output = tf.identity(y_hat, name="output")
        # self.output = tf.identity(output, name="output")

        return self.output

    def loss(self, output, y_placeholder):
        with tf.name_scope("loss"):
            diff = tf.subtract(output, y_placeholder, "diff")

            self.mse = tf.reduce_mean(tf.square(diff, name="diff_square"), name="mse")
            self.image_loss = tf.identity(self.mse, name="image_loss")

            weight_decay_loss = tf.losses.get_regularization_loss()
            self.loss = self.image_loss + weight_decay_loss

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("weight_decay", weight_decay_loss)

            return self.loss

    def summary(self, output, labels):
        tf.summary.image("output_image", output)
        tf.summary.image("grand_truth", labels)
        return super().summary(output, labels)
    
    def post_process(self, input_image, output_image):
        with tf.name_scope("post_process"):
            input_shape = tf.shape(input_image)
            height = input_shape[0]
            width = input_shape[1]
            resized_input_image = tf.image.resize_images(
                input_image,
                [width * 2, height * 2],
                method=tf.image.ResizeMethod.BICUBIC
            )
            # resized_yuv_image = tf.image.rgb_to_yuv(resized_input_image)
            # output = 
            # return resized_input_image
            # import pdb; pdb.set_trace()
            return output_image



            # # return output
            # return output_image
    
    def metrics(self, output, labels):
        output_transposed = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])

        results = {}
        updates = []
        with tf.name_scope('metrics_cals'):
            mean_squared_error, mean_squared_error_update = tf.metrics.mean_squared_error(
                labels,
                output_transposed,
            )
            results["mean_squared_error"] = mean_squared_error
            updates.append(mean_squared_error_update)

            psnr_array = tf.image.psnr(labels, output, max_val=255)
            psnr, psnr_update = tf.metrics.mean(psnr_array)
            results["psnr"] = psnr
            updates.append(psnr_update)

            ssim_array = tf.image.ssim(labels, output, max_val=255)
            ssim, ssim_update = tf.metrics.mean(ssim_array)
            results["ssim"] = ssim
            updates.append(ssim_update)

            # merge all updates
            updates_op = tf.group(*updates)

            return results, updates_op
