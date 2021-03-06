import tensorflow as tf

from util import build_inputs, columns, compile_train_evaluate_and_showcase, base_feature_keys

inputs = build_inputs('wide_n_deep')

# cross feature between current movie and user historical movie
rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([columns['catMovieId'], rated_movie], 10000))

# wide and deep model architecture
# deep part for all input features
feature_columns = [columns[k] for k in base_feature_keys]
deep = tf.keras.layers.DenseFeatures(feature_columns)(inputs)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
# wide part for cross feature
wide = tf.keras.layers.DenseFeatures([crossed_feature])(inputs)
both = tf.keras.layers.concatenate([deep, wide])
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, outputs)

compile_train_evaluate_and_showcase(model)
