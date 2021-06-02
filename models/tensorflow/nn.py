import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10, time_length=5):
        super(Attention, self).__init__()
        self.repeat_vector_time = tf.keras.layers.RepeatVector(time_length)
        self.repeat_vector_emb = tf.keras.layers.RepeatVector(embedding_size)
        self.multiply = tf.keras.layers.Multiply()
        self.dense_32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.dense_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.flatten = tf.keras.layers.Flatten()
        self.permute = tf.keras.layers.Permute((2, 1))

    def call(self, inputs):
        candidate_inputs, gru_hidden_state = inputs
        repeated_candidate_layer = self.repeat_vector_time(candidate_inputs)
        # activation_product_layer shape (batch_size, time_length, embedding_size)
        activation_product_layer = self.multiply([gru_hidden_state, repeated_candidate_layer])
        activation_unit = self.dense_32(activation_product_layer)
        activation_unit = self.dense_1(activation_unit)
        # attention shape (batch_size, time_length)
        attention = tf.squeeze(activation_unit, axis=2)
        attention = self.repeat_vector_emb(attention)
        # attention shape (batch_size, time_length, embedding_size)
        attention = self.permute(attention)
        return attention


class GRUGateParameter(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10):
        super(GRUGateParameter, self).__init__()
        self.multiply = tf.keras.layers.Multiply()
        self.dense_sigmoid = tf.keras.layers.Dense(embedding_size, activation='sigmoid')
        self.dense_tanh = tf.keras.layers.Dense(embedding_size, activation='tanh')
        self.input_w = tf.keras.layers.Dense(embedding_size)
        self.hidden_w = tf.keras.layers.Dense(embedding_size, use_bias=False)

    def call(self, inputs, z=None):
        gru_inputs, hidden_inputs = inputs
        if z is None:
            return self.dense_sigmoid(self.input_w(gru_inputs) + self.hidden_w(hidden_inputs))
        else:
            return self.dense_tanh(self.input_w(gru_inputs) + self.hidden_w(self.multiply([hidden_inputs, z])))


class AUGRU(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10, time_length=5):
        super(AUGRU, self).__init__()
        self.time_length = time_length
        self.embedding_size = embedding_size
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        self.r_gate = GRUGateParameter()
        self.z_gate = GRUGateParameter()
        self.h_gate = GRUGateParameter()

    def call(self, inputs):
        gru_hidden_state_inputs, attention_s = inputs
        initializer = tf.keras.initializers.GlorotUniform()
        hidden_state = initializer(shape=(1, self.embedding_size))
        for t in range(self.time_length):            
            r = self.r_gate([gru_hidden_state_inputs[:, t, :],  hidden_state])
            z = self.z_gate([gru_hidden_state_inputs[:, t, :],  hidden_state])
            h = self.h_gate([gru_hidden_state_inputs[:, t, :],  hidden_state], z)
            attention = self.multiply([attention_s[:, t, :], r])
            
            hidden_state = self.add([self.multiply([(1 - attention), hidden_state]), self.multiply([attention, h])])

        return hidden_state


class AuxiliaryLossLayer(tf.keras.layers.Layer):
    def __init__(self, time_length=5):
        super(AuxiliaryLossLayer, self).__init__()
        self.time_len = time_length - 1
        self.dense_sigmoid_positive_32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.dense_sigmoid_positive_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense_sigmoid_negitive_32 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.dense_sigmoid_negitive_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dot = tf.keras.layers.Dot(axes=(1, 1))
        self.acc = tf.keras.metrics.Accuracy(name='accuracy')
        self.auc = tf.keras.metrics.AUC(name='AUC')
        self.pr = tf.keras.metrics.AUC(curve='PR', name='PR')
    
    def call(self, inputs, alpha=0.5):
        negtive_movie_t1, postive_movie_t0, movie_hidden_state, y_true, y_pred = inputs
        positive_concat_layer = tf.keras.layers.concatenate([movie_hidden_state[:, 0:4, :], postive_movie_t0[:, 1:5, :]])
        positive_concat_layer = self.dense_sigmoid_positive_32(positive_concat_layer)
        positive_loss = self.dense_sigmoid_positive_1(positive_concat_layer)
        
        negtive_concat_layer = tf.keras.layers.concatenate([movie_hidden_state[:, 0:4, :], negtive_movie_t1[:, :, :]])
        negtive_concat_layer = self.dense_sigmoid_negitive_32(negtive_concat_layer)
        negtive_loss = self.dense_sigmoid_negitive_1(negtive_concat_layer)
        # auxiliary_loss_values shape (batch_size, 4, 1)
        auxiliary_loss_values = positive_loss + negtive_loss
        
        final_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + alpha * tf.reduce_mean(tf.reduce_sum(auxiliary_loss_values, axis=1))
        self.add_loss(final_loss, inputs=True)
        self.acc.update_state(y_true, y_pred)
        self.auc.update_state(y_true, y_pred)
        self.pr.update_state(y_true, y_pred)
        # self.add_metric(self.auc.result(), aggregation="mean", name="auc")
        return final_loss
