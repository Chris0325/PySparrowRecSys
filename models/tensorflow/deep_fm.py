import tensorflow as tf

from util import get_sample_datasets, build_inputs, columns, evaluate_and_showcase

inputs = build_inputs('deep_fm')
train_dataset, test_dataset = get_sample_datasets()

# fm first-order term columns: without embedding and concatenate to the output layer directly
fm_first_order_columns = [columns['indMovieId'], columns['indUserId'], columns['indUserGenre1'], columns['indMovieGenre1']]

deep_feature_columns = [tf.feature_column.numeric_column('releaseYear'),
                        tf.feature_column.numeric_column('movieRatingCount'),
                        tf.feature_column.numeric_column('movieAvgRating'),
                        tf.feature_column.numeric_column('movieRatingStddev'),
                        tf.feature_column.numeric_column('userRatingCount'),
                        tf.feature_column.numeric_column('userAvgRating'),
                        tf.feature_column.numeric_column('userRatingStddev'),
                        columns['movieId'],
                        columns['userId']]

item_emb_layer = tf.keras.layers.DenseFeatures([columns['movieId']])(inputs)
user_emb_layer = tf.keras.layers.DenseFeatures([columns['userId']])(inputs)
item_genre_emb_layer = tf.keras.layers.DenseFeatures([columns['movieGenre1']])(inputs)
user_genre_emb_layer = tf.keras.layers.DenseFeatures([columns['userGenre1']])(inputs)

# The first-order term in the FM layer
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
# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

evaluate_and_showcase(model, test_dataset)
