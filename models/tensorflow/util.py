import os
import random
import pandas as pd
import tensorflow as tf

import conf

# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

genre_features = {
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}

# movie id embedding feature
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
movie_ind_col = tf.feature_column.indicator_column(movie_col)  # dense movie id indicator column

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_ind_col = tf.feature_column.indicator_column(user_col)  # dense, user id indicator column

columns = {
    'userId': user_emb_col,
    'indUserId': user_ind_col,
    'movieId': movie_emb_col,
    'catMovieId': movie_col,
    'indMovieId': movie_ind_col,
    'label': tf.feature_column.numeric_column(key='label', default_value=0),
}
# genre embedding features
for feature, vocab in genre_features.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(cat_col, 10)
    ind_col = tf.feature_column.indicator_column(cat_col)  # dense indicator column
    columns[feature] = emb_col
    columns['ind' + feature[0].upper() + feature[1:]] = ind_col

common_numeric_keys = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev', 'userRatingCount',
                       'userAvgRating', 'userRatingStddev']

recent_rate_keys = ['userRatedMovie' + str(i) for i in range(1, 6)]

negtive_movie_keys = ['negtive_userRatedMovie' + str(i) for i in range(2, 6)]

base_feature_keys = common_numeric_keys + list(genre_features.keys()) + ['movieId', 'userId']

# numerical features
for k in common_numeric_keys + recent_rate_keys + negtive_movie_keys:
    columns[k] = tf.feature_column.numeric_column(k)


# define input for keras model
def build_inputs(task):
    inputs = {
        'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
        'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
        'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
        'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
        'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
        'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
        'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

        'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
        'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
        'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),
        'userRatedMovie2': tf.keras.layers.Input(name='userRatedMovie2', shape=(), dtype='int32'),
        'userRatedMovie3': tf.keras.layers.Input(name='userRatedMovie3', shape=(), dtype='int32'),
        'userRatedMovie4': tf.keras.layers.Input(name='userRatedMovie4', shape=(), dtype='int32'),
        'userRatedMovie5': tf.keras.layers.Input(name='userRatedMovie5', shape=(), dtype='int32'),

        'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
        'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
        'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
        'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
        'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
        'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
        'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
        'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),

        'negtive_userRatedMovie2': tf.keras.layers.Input(name='negtive_userRatedMovie2', shape=(), dtype='int32'),
        'negtive_userRatedMovie3': tf.keras.layers.Input(name='negtive_userRatedMovie3', shape=(), dtype='int32'),
        'negtive_userRatedMovie4': tf.keras.layers.Input(name='negtive_userRatedMovie4', shape=(), dtype='int32'),
        'negtive_userRatedMovie5': tf.keras.layers.Input(name='negtive_userRatedMovie5', shape=(), dtype='int32'),

        'label': tf.keras.layers.Input(name='label', shape=(), dtype='int32')
    }

    if task == 'embedding_mlp':
        return {k: inputs[k] for k in base_feature_keys}
    elif task in ['wide_n_deep', 'deep_fm', 'deep_fm_v2']:
        return {k: inputs[k] for k in base_feature_keys + ['userRatedMovie1']}
    elif task == 'neural_cf':
        return {k: inputs[k] for k in ['movieId', 'userId']}
    elif task == 'din':
        return {k: inputs[k] for k in base_feature_keys + recent_rate_keys}
    elif task == 'dien':
        return {k: inputs[k] for k in base_feature_keys + recent_rate_keys + negtive_movie_keys + ['label']}


def get_sample_datasets(batch_size=16, dien=False):
    if dien:
        train_dataset = get_dataset_with_negtive_movie(
            os.path.join(conf.data_directory, "sampledata", "trainingSamples.csv"), batch_size, seed_num=2020)
        test_dataset = get_dataset_with_negtive_movie(
            os.path.join(conf.data_directory, "sampledata", "testSamples.csv"), batch_size, seed_num=2021)
    else:
        train_dataset = tf.data.experimental.make_csv_dataset(os.path.join(conf.data_directory, "sampledata", "trainingSamples.csv"),
                                                              batch_size=batch_size, label_name='label', na_value="0",
                                                              num_epochs=1, ignore_errors=True)

        test_dataset = tf.data.experimental.make_csv_dataset(os.path.join(conf.data_directory, "sampledata", "testSamples.csv"),
                                                              batch_size=batch_size, label_name='label', na_value="0",
                                                              num_epochs=1, ignore_errors=True)

    return train_dataset, test_dataset


def get_dataset_with_negtive_movie(path, batch_size, seed_num):
    tmp_df = pd.read_csv(path)
    tmp_df.fillna(0, inplace=True)
    random.seed(seed_num)
    negtive_movie_df = tmp_df.loc[:, 'userRatedMovie2':'userRatedMovie5'].applymap(lambda x: random.sample(set(range(0, 1001)) - set([int(x)]), 1)[0])
    negtive_movie_df.columns = ['negtive_userRatedMovie2', 'negtive_userRatedMovie3', 'negtive_userRatedMovie4',
                                'negtive_userRatedMovie5']
    tmp_df = pd.concat([tmp_df, negtive_movie_df], axis=1)

    for i in tmp_df.select_dtypes('O').columns:
        tmp_df[i] = tmp_df[i].astype('str')

    if tf.__version__ < '2.3.0':
        tmp_df = tmp_df.sample(n=batch_size * (len(tmp_df) // batch_size), random_state=seed_num)

    dataset = tf.data.Dataset.from_tensor_slices((dict(tmp_df)))
    dataset = dataset.batch(batch_size)
    return dataset


def compile_train_evaluate_and_showcase(model, epochs=5, dien=False):
    train_dataset, test_dataset = get_sample_datasets(dien=dien)
    # compile the model, set loss function, optimizer and evaluation metrics
    if dien:
        model.compile(optimizer='adam')
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

    # train the model
    model.fit(train_dataset, epochs=epochs)

    # evaluate the model
    if dien:
        test_loss, test_roc_auc = model.evaluate(test_dataset)
        print('\n\nTest Loss {},  Test ROC AUC {},'.format(test_loss, test_roc_auc))
    else:
        test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
        print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy, test_roc_auc, test_pr_auc))

    # print some predict results
    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]), " | Actual rating label: ", ("Good Rating" if bool(goodRating) else "Bad Rating"))
