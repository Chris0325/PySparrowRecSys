import os
import tensorflow as tf

import conf
from util import get_sample_datasets, columns, build_inputs, evaluate_and_showcase

inputs = build_inputs('neural_cf')
train_dataset, test_dataset = get_sample_datasets()


# neural cf model arch one. only embedding in each tower, then MLP as the interaction layers
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# neural cf model arch two. embedding+MLP in each tower, then dot product layer as the output
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# neural cf model architecture
model = neural_cf_model_1(inputs, [columns['movieId']], [columns['userId']], [10, 10])

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

evaluate_and_showcase(model, test_dataset)

tf.keras.models.save_model(
    model,
    os.path.join(conf.data_directory, "modeldata", "neuralcf", "002"),
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
