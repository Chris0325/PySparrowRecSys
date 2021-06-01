import tensorflow as tf

from util import build_inputs, columns, compile_train_evaluate_and_showcase, common_numeric_keys

inputs = build_inputs('deep_fm')

# FM first-order term columns: without embedding and concatenate to the output layer directly
fm_first_order_columns = [columns['indMovieId'], columns['indUserId'], columns['indUserGenre1'], columns['indMovieGenre1']]

deep_feature_columns = [columns[k] for k in common_numeric_keys + ['movieId', 'userId']]

item_emb_layer = tf.keras.layers.DenseFeatures([columns['movieId']])(inputs)
user_emb_layer = tf.keras.layers.DenseFeatures([columns['userId']])(inputs)
item_genre_emb_layer = tf.keras.layers.DenseFeatures([columns['movieGenre1']])(inputs)
user_genre_emb_layer = tf.keras.layers.DenseFeatures([columns['userGenre1']])(inputs)

# first-order term in the FM layer
fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)

# FM part, cross different categorical feature embeddings
product_layer_item_user = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_emb_layer])
product_layer_item_genre_user_genre = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_genre_emb_layer])
product_layer_item_genre_user = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_emb_layer])
product_layer_user_genre_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_genre_emb_layer])

# deep part, MLP to generalize all input features
deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)

# concatenate fm part and deep part
concat_layer = tf.keras.layers.concatenate([fm_first_order_layer, product_layer_item_user, product_layer_item_genre_user_genre,
                                            product_layer_item_genre_user, product_layer_user_genre_item, deep], axis=1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

model = tf.keras.Model(inputs, output_layer)

compile_train_evaluate_and_showcase(model)
