import tensorflow as tf

from util import build_inputs, columns, common_numeric_keys, GENRE_FEATURES, compile_train_evaluate_and_showcase

inputs = build_inputs('embedding_mlp')

# embedding + MLP model architecture
feature_columns = [columns[k] for k in common_numeric_keys + list(GENRE_FEATURES.keys()) + ['movieId', 'userId']]
x = tf.keras.layers.DenseFeatures(feature_columns)(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

compile_train_evaluate_and_showcase(model)
