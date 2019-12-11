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

        self.H = []
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
        input_feature_num,
        output_feature_num,
        is_training,
        use_batch_norm=False,
        dropout_rate=1.0,
    ):
        with tf.variable_scope(name):
            shape_of_weight = [
                kernel_size,
                kernel_size,
                input_feature_num,
                output_feature_num,
            ]
            w = self._weight(shape=shape_of_weight, name="conv_W")

            shape_of_bias = [output_feature_num]
            b = self._bias(shape=shape_of_bias, name="conv_B")

            z = self._conv2d(
                input, w, stride=1, bias=b, use_batch_norm=use_batch_norm, name=name
            )
            
            a = self._prelu(z, output_feature_num, name=name)
            
            if dropout_rate < 1.0:
                a = tf.nn.dropout(a, dropout_rate, name="dropout")

            self.H.append(a)

            # Save image
            # shapes = w.get_shape().as_list()
            # weights = tf.reshape(w, [shapes[0], shapes[1], shapes[2] * shapes[3]])
            # weights_transposed = tf.transpose(weights, [2, 0, 1])
            # weights_transposed = tf.reshape(
            #     weights_transposed, [shapes[2] * shapes[3], shapes[0], shapes[1], 1]
            # )
            # tf.summary.image("weights", weights_transposed, max_outputs=6)

        self.Weights.append(w)
        self.Biases.append(b)

        return a

    def _pixel_shuffler(
        self, name, input, kernel_size, scale, input_feature_num, output_feature_num, is_training
    ):
        with tf.variable_scope(name):
            self._convolutional_block(
                name + "_CNN",
                input,
                kernel_size,
                input_feature_num=input_feature_num,
                output_feature_num=scale * scale * output_feature_num,
                use_batch_norm=False,
                is_training=is_training
            )

            self.H.append(tf.depth_to_space(self.H[-1], scale))

    def placeholders(self):
        x = tf.placeholder(
            tf.float32, shape=[None, None, None, self.input_channel], name="x"
        )
        y = tf.placeholder(
            tf.float32, shape=[None, None, None, self.output_channel], name="y"
        )
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        return x, y

    def calc_filters(self, first, last, layers, decay):
        return [
            int((first - last) * (1 - pow(i / float(layers - 1), 1.0 / decay)) + last)
            for i in range(layers)
        ]
        

    def base(self, x, is_training):
        tf.summary.image("input_image", x)

        # building feature extraction layers
        output_feature_num = self.filters
        total_output_feature_num = 0
        input_feature_num = self.input_channel
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

        for i in range(self.layers):
            if self.min_filters != 0 and i > 0:
                x1 = i / float(self.layers - 1)
                y1 = pow(x1, 1.0 / self.filters_decay_gamma)
                output_feature_num = int(
                    (self.filters - self.min_filters) * (1 - y1) + self.min_filters
                )

                print(
                    "x1, {}, y1, {}, output_feature_num: {}".format(
                        x1, y1, output_feature_num
                    )
                )
            self._convolutional_block(
                "CNN%d" % (i + 1),
                input_tensor,
                kernel_size=3,
                input_feature_num=input_feature_num,
                output_feature_num=output_feature_num,
                use_batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                is_training=is_training
            )

            input_feature_num = output_feature_num
            input_tensor = self.H[-1]
            total_output_feature_num += output_feature_num

        with tf.variable_scope("Concat"):
            self.H_concat = tf.concat(self.H, 3, name="H_concat")

        # building reconstruction layers
        self._convolutional_block(
            "A1",
            self.H_concat,
            kernel_size=1,
            input_feature_num=total_output_feature_num,
            output_feature_num=64,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )

        self._convolutional_block(
            "B1",
            self.H_concat,
            kernel_size=1,
            input_feature_num=total_output_feature_num,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        self._convolutional_block(
            "B2",
            self.H[-1],
            kernel_size=3,
            input_feature_num=32,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            is_training=is_training
        )
        self.H.append(tf.concat([self.H[-1], self.H[-3]], 3, name="Concat2"))

        # building upsampling layer
        pixel_shuffler_channel = 64 + 32
        self._pixel_shuffler(
            "Up-PS",
            self.H[-1],
            kernel_size=3,
            scale=self.scale,
            input_feature_num=pixel_shuffler_channel,
            output_feature_num=pixel_shuffler_channel,
            is_training=is_training
        )

        self._convolutional_block(
            "R-CNN0",
            self.H[-1],
            kernel_size=3,
            input_feature_num=pixel_shuffler_channel,
            output_feature_num=self.output_channel,
            is_training=is_training
        )

        y_hat = tf.add(self.H[-1], x2, name="output")

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

            l2_norm_losses = [tf.nn.l2_loss(w) for w in self.Weights]
            l2_norm_loss = self.weight_decay_rate + tf.add_n(l2_norm_losses)
            self.loss = self.image_loss + l2_norm_loss

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("weight_decay", l2_norm_loss)

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
