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


class Dcscn(BaseNetwork):
    """ """

    def __init__(
        self,
        input_channel=1,
        output_channel=1,
        weight_decay_rate=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_channel = input_channel
        self.output_channel = output_channel

        # L2 decay
        self.weight_decay_rate = weight_decay_rate

        # Scale factor for Super Resolution (should be 2 or more)
        self.scale = 2

        # Number of feature extraction layers
        self.layers = 12

        # Number of filters of first feature-extraction CNNs
        self.filters = 196

        # Number of filters of last feature-extraction CNNs
        self.min_filters = 48

        # Number of CNN filters are decayed from [filters] to [min_filters] by this gamma
        self.filters_decay_gamma = 1.5

        # Output nodes should be kept by this probability. If 1, don't use dropout.
        self.dropout_rate = 0.8

        # Use batch normalization after each CNNs
        self.batch_norm = False

        self.H = []
        self.Weights = []
        self.Biases = []

        self.mse = None
        self.image_loss = None

    def _he_initializer(self, shape):
        n = shape[0] * shape[1] * shape[2]
        stddev = math.sqrt(2.0 / n)
        return tf.truncated_normal(shape=shape, stddev=stddev)

    def _weight(self, shape, name="weight"):
        initial = self._he_initializer(shape)
        return tf.Variable(initial, name=name)

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
        use_batch_norm=False,
        dropout_rate=1.0,
        dropout=None,
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

            if dropout_rate < 1.0:
                z = tf.nn.dropout(z, dropout, name="dropout")

            a = self._prelu(z, output_feature_num, name=name)

            self.H.append(a)

            # Save image
            shapes = w.get_shape().as_list()
            weights = tf.reshape(w, [shapes[0], shapes[1], shapes[2] * shapes[3]])
            weights_transposed = tf.transpose(weights, [2, 0, 1])
            weights_transposed = tf.reshape(
                weights_transposed, [shapes[2] * shapes[3], shapes[0], shapes[1], 1]
            )
            tf.summary.image("weights", weights_transposed, max_outputs=6)

        self.Weights.append(w)
        self.Biases.append(b)

        return a

    def _pixel_shuffler(
        self, name, input, kernel_size, scale, input_feature_num, output_feature_num
    ):
        with tf.variable_scope(name):
            self._convolutional_block(
                name + "_CNN",
                input,
                kernel_size,
                input_feature_num=input_feature_num,
                output_feature_num=scale * scale * output_feature_num,
                use_batch_norm=False,
            )

            self.H.append(tf.depth_to_space(self.H[-1], scale))

    def placeholders(self):
        x = tf.placeholder(
            tf.float32, shape=[None, None, None, self.input_channel], name="x"
        )
        y = tf.placeholder(
            tf.float32, shape=[None, None, None, self.output_channel], name="y"
        )
        self.x2 = tf.placeholder(
            tf.float32, shape=[None, None, None, self.output_channel], name="x2"
        )
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout_keep_rate")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        return x, y

    def forward(self, x, x2, dropout):
        # building feature extraction layers
        output_feature_num = self.filters
        total_output_feature_num = 0
        input_feature_num = self.input_channel
        input_tensor = x

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
                dropout=dropout,
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
            dropout=dropout,
        )

        self._convolutional_block(
            "B1",
            self.H_concat,
            kernel_size=1,
            input_feature_num=total_output_feature_num,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            dropout=dropout,
        )
        self._convolutional_block(
            "B2",
            self.H[-1],
            kernel_size=3,
            input_feature_num=32,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            dropout=dropout,
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
        )

        self._convolutional_block(
            "R-CNN0",
            self.H[-1],
            kernel_size=3,
            input_feature_num=pixel_shuffler_channel,
            output_feature_num=self.output_channel,
        )

        y_hat = tf.add(self.H[-1], x2, name="output")

        with tf.name_scope("Y_"):
            mean = tf.reduce_mean(y_hat)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(y_hat - mean)))
            tf.summary.scalar("output/mean", mean)
            tf.summary.scalar("output/stddev", stddev)
            tf.summary.histogram("output", y_hat)

        return y_hat

    def inference(self, x_placeholder, is_training):
        y_hat = self.forward(x_placeholder, self.x2, self.dropout)
        self.output = tf.identity(y_hat, name="output")

        return self.output

    def optimizer(self, global_step):
        beta1 = 0.9
        beta2 = 0.999
        if self.batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(
                    self.learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-8
                )
        else:
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-8
            )
        
        return optimizer

    def loss(self, output, y_placeholder):
        diff = tf.subtract(self.output, y_placeholder)

        self.mse = tf.reduce_mean(tf.square(diff, name="diff_square"), name="mse")
        self.image_loss = tf.identity(self.mse, name="image_loss")

        l2_norm_losses = [tf.nn.l2_loss(w) for w in self.Weights]
        l2_norm_loss = self.weight_decay_rate + tf.add_n(l2_norm_losses)
        self.loss = self.image_loss + l2_norm_loss

        tf.summary.scalar("loss", self.loss)

        return self.loss
    
    def metrics(self, output, y_placeholder):
        with tf.name_scope("metrics_calc"):
            y_placeholder = tf.cast(y_placeholder, tf.float32)

            if self.is_debug:
                y_placeholder = tf.Print(y_placeholder, [tf.shape(y_placeholder), tf.argmax(y_placeholder, 1)], message="y_placeholder:", summarize=200)
                output = tf.Print(output,
                                   [tf.shape(output), tf.argmax(output, 1)], message="output:", summarize=200)

            accuracy, accuracy_update = self._calc_top_k(output, y_placeholder, k=1)

            if(self.num_classes > 3):
                accuracy_top3, accuracy_top3_update = self._calc_top_k(output, y_placeholder, k=3)
            else:
                accuracy_top3, accuracy_top3_update = tf.compat.v1.metrics.mean(tf.ones(self.batch_size))

            if(self.num_classes > 5):
                accuracy_top5, accuracy_top5_update = self._calc_top_k(output, y_placeholder, k=5)
            else:
                accuracy_top5, accuracy_top5_update = tf.compat.v1.metrics.mean(tf.ones(self.batch_size))

            updates = tf.group(accuracy_update, accuracy_top3_update, accuracy_top5_update)

        metrics_dict = {"accuracy": accuracy, "accuracy_top3": accuracy_top3, "accuracy_top5": accuracy_top5}
        return metrics_dict, updates

    def _calc_top_k(self, softmax, labels, k):
        """Calculate the mean top k accuracy.
        In the case that multiple classes are on the top k boundary, the order of the class indices is used
        to break the tie - lower indices given preference - so that only k predictions are included in the top k.

        Args:
            softmax (Tensor): class predictions from the softmax. Shape is [batch_size, num_classes].
            labels (Tensor): onehot ground truth labels. Shape is [batch_size, num_classes].
            k (Int): number of top predictions to use.

        """

        argmax_labels = tf.cast(tf.argmax(labels, 1), tf.int32)
        argmax_labels = tf.expand_dims(argmax_labels, 1)
        _, top_predicted_indices = tf.nn.top_k(softmax, k)
        accuracy_topk, accuracy_topk_update = tf.compat.v1.metrics.mean(
            tf.cast(tf.reduce_any(tf.equal(top_predicted_indices, argmax_labels), axis=1), tf.float32)
        )
        return accuracy_topk, accuracy_topk_update