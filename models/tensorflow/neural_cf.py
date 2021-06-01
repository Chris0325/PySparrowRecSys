import os
import tensorflow as tf

import conf
from util import columns, build_inputs, compile_train_evaluate_and_showcase

inputs = build_inputs('neural_cf')


# neural cf model arch one. only embedding in each tower, then MLP as the interaction layers
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    neural_cf_model = tf.keras.Model(feature_inputs, outputs)
    return neural_cf_model


# neural cf model arch two. embedding+MLP in each tower, then dot product layer as the output
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    x = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    neural_cf_model = tf.keras.Model(feature_inputs, outputs)
    return neural_cf_model


# neural cf model architecture
model = neural_cf_model_1(inputs, [columns['movieId']], [columns['userId']], [10, 10])

compile_train_evaluate_and_showcase(model)

tf.keras.models.save_model(model, os.path.join(conf.data_directory, "modeldata", "neuralcf", "002"))
