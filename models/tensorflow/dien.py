"""
Author:
    zcxia23, 854695170@qq.com

Diff with DIN:
    1、GRU with attentional update gate (AUGRU) 
    2、auxiliary loss function with click or not click  movie(negetive sampleming)

Reference:
    [1] Zhou G ,  Mou N ,  Fan Y , et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J].  2018.
"""

import tensorflow as tf

from nn import Attention, AUGRU, AuxiliaryLossLayer
from util import build_inputs, columns, recent_rate_keys, negtive_movie_keys, compile_train_evaluate_and_showcase, \
    user_profile_keys, context_keys

inputs = build_inputs('dien')

candidate_movie_col = [tf.feature_column.numeric_column(key='movieId', default_value=0)]

# user behaviors
recent_rate_col = [columns[k] for k in recent_rate_keys]

negtive_movie_col = [columns[k] for k in negtive_movie_keys]

# user profile
user_profile = [columns[k] for k in user_profile_keys]

# context features
context_features = [columns[k] for k in context_keys]

candidate_layer = tf.keras.layers.DenseFeatures(candidate_movie_col)(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
negtive_movie_layer = tf.keras.layers.DenseFeatures(negtive_movie_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)
y_true = tf.keras.layers.DenseFeatures([columns['label']])(inputs)

# activation unit
movie_emb_layer = tf.keras.layers.Embedding(input_dim=1001, output_dim=10, mask_zero=True)  # mask zero

user_behaviors_emb_layer = movie_emb_layer(user_behaviors_layer)
candidate_emb_layer = movie_emb_layer(candidate_layer) 
negtive_movie_emb_layer = movie_emb_layer(negtive_movie_layer) 

# candidate_emb_layer shape (batch_size, 10)
candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)

# user_behaviors_hidden_state shape (batch_size, 5, 10)
user_behaviors_hidden_state = tf.keras.layers.GRU(10, return_sequences=True)(user_behaviors_emb_layer)

attention_score = Attention()([candidate_emb_layer, user_behaviors_hidden_state])

augru_emb = AUGRU()([user_behaviors_hidden_state, attention_score])

concat_layer = tf.keras.layers.concatenate([augru_emb,  candidate_emb_layer, user_profile_layer, context_features_layer])

output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
y_pred = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

loss = AuxiliaryLossLayer()([negtive_movie_emb_layer, user_behaviors_emb_layer, user_behaviors_hidden_state, y_true, y_pred])

model = tf.keras.Model(inputs=inputs, outputs=[y_pred, loss])

compile_train_evaluate_and_showcase(model, dien=True)
