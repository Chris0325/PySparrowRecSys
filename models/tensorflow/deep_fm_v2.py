"""
Diff with DeepFM:
    1. separate categorical features from dense features when processing first order features and second order features
    2. modify original fm part with a fully crossed fm part
"""

import tensorflow as tf

from util import build_inputs, columns, common_numeric_keys, compile_train_evaluate_and_showcase

inputs = build_inputs('deep_fm_v2')

# FM first-order categorical items
cat_columns = [columns['indMovieId'], columns['indUserId'], columns['indUserGenre1'], columns['indMovieGenre1']]

deep_columns = [columns[k] for k in common_numeric_keys]

first_order_cat_feature = tf.keras.layers.DenseFeatures(cat_columns)(inputs)
first_order_cat_feature = tf.keras.layers.Dense(1)(first_order_cat_feature)
first_order_deep_feature = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
first_order_deep_feature = tf.keras.layers.Dense(1)(first_order_deep_feature)

# first order feature shape (batch_size, 1)
first_order_feature = first_order_cat_feature + first_order_deep_feature


second_order_cat_columns_emb = [tf.keras.layers.DenseFeatures([columns[k]])(inputs) for k in ['movieGenre1', 'movieId', 'userGenre1', 'userId']]

second_order_cat_columns = []
for feature_emb in second_order_cat_columns_emb:
    feature = tf.keras.layers.Dense(64)(feature_emb)
    feature = tf.keras.layers.Reshape((-1, 64))(feature)
    second_order_cat_columns.append(feature)

second_order_deep_columns = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
second_order_deep_columns = tf.keras.layers.Dense(64)(second_order_deep_columns)
second_order_deep_columns = tf.keras.layers.Reshape((-1, 64))(second_order_deep_columns)

# second_order_fm_feature shape (batch_size, 11, 64)
second_order_fm_feature = tf.keras.layers.concatenate(second_order_cat_columns + [second_order_deep_columns], axis=1)

deep_feature = tf.keras.layers.flatten(second_order_fm_feature)
deep_feature = tf.keras.layers.Dense(32, activation='relu')(deep_feature)
deep_feature = tf.keras.layers.Dense(16, activation='relu')(deep_feature)

# second_order_sum_feature shape (batch_size, 64), various features have been merged for each embedding position
second_order_sum_feature = tf.reduce_sum(second_order_fm_feature, axis=1)
second_order_sum_square_feature = tf.keras.layers.multiply([second_order_sum_feature, second_order_sum_feature])
second_order_square_feature = tf.keras.layers.multiply([second_order_fm_feature, second_order_fm_feature])
second_order_square_sum_feature = tf.reduce_sum(second_order_square_feature, axis=1)

# second_order_fm_feature
second_order_fm_feature = tf.keras.layers.subtract([second_order_sum_square_feature, second_order_square_sum_feature])

concatenated_outputs = tf.keras.layers.concatenate([first_order_feature, second_order_fm_feature, deep_feature])
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated_outputs)

model = tf.keras.Model(inputs, outputs)

compile_train_evaluate_and_showcase(model)
