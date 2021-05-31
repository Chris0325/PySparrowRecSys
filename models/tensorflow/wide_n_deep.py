import tensorflow as tf

from util import get_sample_datasets, GENRE_FEATURES, inputs, columns, common_numeric_keys

train_dataset, test_dataset = get_sample_datasets()

# cross feature between current movie and user historical movie
rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([columns['movieId'], rated_movie], 10000))

# wide and deep model architecture
# deep part for all input features
feature_columns = [columns[k] for k in common_numeric_keys + list(GENRE_FEATURES.keys()) + ['movieId', 'userId']]
deep = tf.keras.layers.DenseFeatures(feature_columns)(inputs)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
# wide part for cross feature
wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)
both = tf.keras.layers.concatenate([deep, wide])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, output_layer)

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]), " | Actual rating label: ", ("Good Rating" if bool(goodRating) else "Bad Rating"))
