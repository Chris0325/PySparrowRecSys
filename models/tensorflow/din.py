import tensorflow as tf

from util import build_inputs, columns, recent_rate_keys, compile_train_evaluate_and_showcase

inputs = build_inputs('din')

candidate_movie_col = [tf.feature_column.numeric_column(key='movieId', default_value=0)]

# user profile
user_profile = [columns[k] for k in ['userId', 'userGenre1', 'userRatingCount', 'userAvgRating', 'userRatingStddev']]

# context features
context_features = [columns[k] for k in ['movieGenre1', 'releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev']]

recent_rate_col = [columns[k] for k in recent_rate_keys]

candidate_layer = tf.keras.layers.DenseFeatures(candidate_movie_col)(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)

# Activation Unit

movie_emb_layer = tf.keras.layers.Embedding(input_dim=1001, output_dim=10, mask_zero=True)  # mask zero

user_behaviors_emb_layer = movie_emb_layer(user_behaviors_layer) 

candidate_emb_layer = movie_emb_layer(candidate_layer) 
candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)

repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(5)(candidate_emb_layer)

activation_sub_layer = tf.keras.layers.Subtract()([user_behaviors_emb_layer, repeated_candidate_emb_layer])  # element-wise sub
activation_product_layer = tf.keras.layers.Multiply()([user_behaviors_emb_layer, repeated_candidate_emb_layer])  # element-wise product

activation_all = tf.keras.layers.concatenate([activation_sub_layer, user_behaviors_emb_layer,
                                              repeated_candidate_emb_layer, activation_product_layer], axis=-1)

activation_unit = tf.keras.layers.Dense(32)(activation_all)
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
activation_unit = tf.keras.layers.Flatten()(activation_unit)
activation_unit = tf.keras.layers.RepeatVector(10)(activation_unit)
activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)
activation_unit = tf.keras.layers.Multiply()([user_behaviors_emb_layer, activation_unit])

# sum pooling
user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_unit)

# fc layer
concat_layer = tf.keras.layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
                                            candidate_emb_layer, context_features_layer])
output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

model = tf.keras.Model(inputs, output_layer)

compile_train_evaluate_and_showcase(model)
