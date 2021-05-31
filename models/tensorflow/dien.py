"""
Author:
    zcxia23, 854695170@qq.com

Diff with DIN:
    1、GRU with attentional update gate (AUGRU) 
    2、auxiliary loss function with click or not click  movie(negetive sampleming)

Reference:
    [1] Zhou G ,  Mou N ,  Fan Y , et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J].  2018.
"""

import os
import pandas as pd
import tensorflow as tf
import random

import conf
from util import build_inputs, columns, recent_rate_keys, negtive_movie_keys, compile_train_evaluate_and_showcase

inputs = build_inputs('dien')

candidate_movie_col = [tf.feature_column.numeric_column(key='movieId', default_value=0)]

# user behaviors
recent_rate_col = [columns[k] for k in recent_rate_keys]

negtive_movie_col = [columns[k] for k in negtive_movie_keys]

# user profile
user_profile = [columns[k] for k in ['userId', 'userGenre1', 'userRatingCount', 'userAvgRating', 'userRatingStddev']]

# context features
context_features = [columns[k] for k in ['movieGenre1', 'releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev']]

candidate_layer = tf.keras.layers.DenseFeatures(candidate_movie_col)(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
negtive_movie_layer = tf.keras.layers.DenseFeatures(negtive_movie_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)
y_true = tf.keras.layers.DenseFeatures([columns['label']])(inputs)

# Activation Unit
movie_emb_layer = tf.keras.layers.Embedding(input_dim=1001, output_dim=10, mask_zero=True)  # mask zero

user_behaviors_emb_layer = movie_emb_layer(user_behaviors_layer) 
candidate_emb_layer = movie_emb_layer(candidate_layer) 
negtive_movie_emb_layer = movie_emb_layer(negtive_movie_layer) 

candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)

user_behaviors_hidden_state=tf.keras.layers.GRU(10, return_sequences=True)(user_behaviors_emb_layer)


class attention(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10, time_length=5, ):
        super().__init__()
        self.time_length = time_length  
        self.embedding_size = embedding_size
        self.RepeatVector_time = tf.keras.layers.RepeatVector(self.time_length)
        self.RepeatVector_emb = tf.keras.layers.RepeatVector(self.embedding_size)        
        self.Multiply = tf.keras.layers.Multiply()
        self.Dense32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.Dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.Flatten = tf.keras.layers.Flatten()
        self.Permute = tf.keras.layers.Permute((2, 1))
        
    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        candidate_inputs, gru_hidden_state = inputs
        repeated_candidate_layer = self.RepeatVector_time(candidate_inputs)
        activation_product_layer = self.Multiply([gru_hidden_state, repeated_candidate_layer])
        activation_unit = self.Dense32(activation_product_layer)
        activation_unit = self.Dense1(activation_unit)  
        Repeat_attention_s = tf.squeeze(activation_unit, axis=2)
        Repeat_attention_s = self.RepeatVector_emb(Repeat_attention_s)
        Repeat_attention_s = self.Permute(Repeat_attention_s)

        return Repeat_attention_s


attention_score = attention()([candidate_emb_layer, user_behaviors_hidden_state])


class GRU_gate_parameter(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10):
        super().__init__()
        self.embedding_size = embedding_size        
        self.Multiply = tf.keras.layers.Multiply()
        self.Dense_sigmoid = tf.keras.layers.Dense(self.embedding_size, activation='sigmoid')
        self.Dense_tanh =tf.keras.layers.Dense(self.embedding_size, activation='tanh')
        
    def build(self, input_shape):
        self.input_w = tf.keras.layers.Dense(self.embedding_size, activation=None, use_bias=True)
        self.hidden_w = tf.keras.layers.Dense(self.embedding_size, activation=None, use_bias=False)

    def call(self, inputs, Z_t_inputs=None):
        gru_inputs, hidden_inputs = inputs
        if Z_t_inputs == None:
            return self.Dense_sigmoid(self.input_w(gru_inputs) + self.hidden_w(hidden_inputs))
        else:           
            return self.Dense_tanh(self.input_w(gru_inputs) + self.hidden_w(self.Multiply([hidden_inputs, Z_t_inputs])))

                                                                                                                                                                
class AUGRU(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10,  time_length=5):
        super().__init__()
        self.time_length = time_length
        self.embedding_size = embedding_size      
        self.Multiply = tf.keras.layers.Multiply()
        self.Add = tf.keras.layers.Add()
    
    def build(self, input_shape):
        self.R_t = GRU_gate_parameter()
        self.Z_t = GRU_gate_parameter()                                                                                     
        self.H_t_next = GRU_gate_parameter()     

    def call(self, inputs ):
        gru_hidden_state_inputs, attention_s = inputs
        initializer = tf.keras.initializers.GlorotUniform()
        AUGRU_hidden_state = tf.reshape(initializer(shape=(1, self.embedding_size)), shape=(-1, self.embedding_size))
        for t in range(self.time_length):            
            r_t = self.R_t([gru_hidden_state_inputs[:, t, :],  AUGRU_hidden_state])
            z_t = self.Z_t([gru_hidden_state_inputs[:, t, :],  AUGRU_hidden_state])
            h_t_next = self.H_t_next([gru_hidden_state_inputs[:, t, :],  AUGRU_hidden_state], z_t)
            Rt_attention = self.Multiply([attention_s[:, t, :], r_t])
            
            AUGRU_hidden_state = self.Add([self.Multiply([(1-Rt_attention), AUGRU_hidden_state]),
                                           self.Multiply([Rt_attention, h_t_next])])

        return AUGRU_hidden_state


augru_emb = AUGRU()([user_behaviors_hidden_state, attention_score])

concat_layer = tf.keras.layers.concatenate([augru_emb,  candidate_emb_layer, user_profile_layer, context_features_layer])

output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
y_pred = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)


class auxiliary_loss_layer(tf.keras.layers.Layer):
    def __init__(self, time_length=5):
        super().__init__()
        self.time_len = time_length-1        
        self.Dense_sigmoid_positive32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.Dense_sigmoid_positive1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.Dense_sigmoid_negitive32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.Dense_sigmoid_negitive1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.Dot = tf.keras.layers.Dot(axes=(1, 1))
        self.auc = tf.keras.metrics.AUC()
        
    def build(self, input_shape):
        pass
    
    def call(self, inputs,alpha=0.5):
        negtive_movie_t1, postive_movie_t0, movie_hidden_state, y_true, y_pred = inputs
        positive_concat_layer = tf.keras.layers.concatenate([movie_hidden_state[:, 0:4, :], postive_movie_t0[:, 1:5, :]])
        positive_concat_layer = self.Dense_sigmoid_positive32(positive_concat_layer)
        positive_loss = self.Dense_sigmoid_positive1(positive_concat_layer)
        
        negtive_concat_layer = tf.keras.layers.concatenate([movie_hidden_state[:, 0:4, :], negtive_movie_t1[:, :, :]])
        negtive_concat_layer = self.Dense_sigmoid_negitive32(negtive_concat_layer)
        negtive_loss = self.Dense_sigmoid_negitive1(negtive_concat_layer)        
        auxiliary_loss_values = positive_loss + negtive_loss
        
        final_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) - alpha * tf.reduce_mean(tf.reduce_sum(auxiliary_loss_values, axis=1))
        self.add_loss(final_loss, inputs=True)
        self.auc.update_state(y_true, y_pred)
        self.add_metric(self.auc.result(), aggregation="mean", name="auc_value")        
        
        return final_loss


auxiliary_loss_value = auxiliary_loss_layer()([negtive_movie_emb_layer, user_behaviors_emb_layer, user_behaviors_hidden_state, y_true, y_pred])

model = tf.keras.Model(inputs=inputs, outputs=[y_pred, auxiliary_loss_value])
model.summary()

compile_train_evaluate_and_showcase(model, dien=True)
