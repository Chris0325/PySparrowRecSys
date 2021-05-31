"""
Diff with DeepFM:
    1. separate categorical features from dense features when processing first order features and second order features
    2. modify original fm part with a fully crossed fm part
"""

import tensorflow as tf

from util import get_sample_datasets, build_inputs, columns, common_numeric_keys, compile_train_evaluate_and_showcase

inputs = build_inputs('deep_fm_v2')
train_dataset, test_dataset = get_sample_datasets()

# fm first-order categorical items
cat_columns = [columns['indMovieId'], columns['indUserId'], columns['indUserGenre1'], columns['indMovieGenre1']]

deep_columns = [columns[k] for k in common_numeric_keys]

first_order_cat_feature = tf.keras.layers.DenseFeatures(cat_columns)(inputs)
first_order_cat_feature = tf.keras.layers.Dense(1, activation=None)(first_order_cat_feature)
first_order_deep_feature = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
first_order_deep_feature = tf.keras.layers.Dense(1, activation=None)(first_order_deep_feature)

# first order feature
first_order_feature = tf.keras.layers.Add()([first_order_cat_feature, first_order_deep_feature])

second_order_cat_columns_emb = [tf.keras.layers.DenseFeatures([columns['movieGenre1']])(inputs),
                                tf.keras.layers.DenseFeatures([columns['movieId']])(inputs),
                                tf.keras.layers.DenseFeatures([columns['userGenre1']])(inputs),
                                tf.keras.layers.DenseFeatures([columns['userId']])(inputs)
                                ]

second_order_cat_columns = []
for feature_emb in second_order_cat_columns_emb:
    feature = tf.keras.layers.Dense(64, activation=None)(feature_emb)
    feature = tf.keras.layers.Reshape((-1, 64))(feature)
    second_order_cat_columns.append(feature)

second_order_deep_columns = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
second_order_deep_columns = tf.keras.layers.Dense(64, activation=None)(second_order_deep_columns)
second_order_deep_columns = tf.keras.layers.Reshape((-1, 64))(second_order_deep_columns)
second_order_fm_feature = tf.keras.layers.Concatenate(axis=1)(second_order_cat_columns + [second_order_deep_columns])

# second_order_deep_feature
deep_feature = tf.keras.layers.Flatten()(second_order_fm_feature)
deep_feature = tf.keras.layers.Dense(32, activation='relu')(deep_feature)
deep_feature = tf.keras.layers.Dense(16, activation='relu')(deep_feature)


class ReduceLayer(tf.keras.layers.Layer):
    def __init__(self, axis, op='sum', **kwargs):
        super().__init__()
        self.axis = axis
        self.op = op
        assert self.op in ['sum', 'mean']

    def build(self, input_shape):
        pass

    def call(self, input, **kwargs):
        if self.op == 'sum':
            return tf.reduce_sum(input, axis=self.axis)
        elif self.op == 'mean':
            return tf.reduce_mean(input, axis=self.axis)
        return tf.reduce_sum(input, axis=self.axis)


second_order_sum_feature = ReduceLayer(1)(second_order_fm_feature)
second_order_sum_square_feature = tf.keras.layers.multiply([second_order_sum_feature, second_order_sum_feature])
second_order_square_feature = tf.keras.layers.multiply([second_order_fm_feature, second_order_fm_feature])
second_order_square_sum_feature = ReduceLayer(1)(second_order_square_feature)
# second_order_fm_feature
second_order_fm_feature = tf.keras.layers.subtract([second_order_sum_square_feature, second_order_square_sum_feature])

concatenated_outputs = tf.keras.layers.Concatenate(axis=1)([first_order_feature, second_order_fm_feature, deep_feature])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated_outputs)

model = tf.keras.Model(inputs, output_layer)

compile_train_evaluate_and_showcase(model, train_dataset, test_dataset)
