import tensorflow as tf

from util import build_inputs, columns, compile_train_evaluate_and_showcase, base_feature_keys

inputs = build_inputs('embedding_mlp')

# embedding + MLP model architecture
feature_columns = [columns[k] for k in base_feature_keys]
x = tf.keras.layers.DenseFeatures(feature_columns)(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

compile_train_evaluate_and_showcase(model)
