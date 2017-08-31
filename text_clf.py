import numpy as np
import os
import warnings

# Scikit-learn classifier and report
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Gensim Word2Vec
from gensim.models import word2vec, KeyedVectors
from gensim.scripts import glove2word2vec

# Keras for Deep Learning
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM, merge
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
os.environ['KERAS_BACKEND'] = 'theano'
reload(K)
np.random.seed(0)  # for equal random state between experiments

# ---------------------- Parameters section -------------------

# Trained corpus: not trained, from the same dataset reviews, from web text or from news text
#                   - web and twitter   : https://nlp.stanford.edu/projects/glove/
#                   - news              : https://code.google.com/archive/p/word2vec/
embeddings_from = 'reviews'  # none | reviews | twitter | news

# Paths:
my_path = './Datasets/'
# corpus
w2v_dir = my_path+'models_w2v'
news_path = my_path+'corpus/GoogleNews-vectors-negative300.bin'
web_path = my_path+'corpus/glove.840B.300d.txt'
twit_path = my_path+'corpus/glove.twitter.27B.100d.txt'
# trained models
cnn_model_path = my_path+'models/cnn_model'
lstm_model_path = my_path+'models/lstm_model'

# CNN Model parameters
update_embeddings = True
batch_size = 32
num_epochs = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# SVM Model parameters
train_w2v = True
mean_embeddings = True

# Training parameters
validation_size = 0.1    # For each epoch validation
test_size = 0.2          # Hold-out test set
class_weights = {0: 1.,  # Compensate imbalanced dataset
                 1: 1.}

# Pre-prossessing parameters
review_length = 300

# Word2Vec parameters (see function train_word2vec)
embedding_dim_rev = 100  # Vector dimension
min_word_count = 1
context = 5              # Number of neighbour words used
num_workers = 2          # Number of threads to run in parallel
downsampling = 1e-3      # Downsample setting for frequent words

# ---------------------- Parameters end -----------------------


def cnn_class(data, labels, vocabulary_inv, classes=['Negative', 'Positive'], override=True, class_weights=class_weights):
    """
    Build a CNN model, train and show prediction results.
    :param data: list of indexed reviews
    :param labels: list of labels
    :param vocabulary_inv: dictionary matching {int:str}
    :param classes: classes names for printing results
    :param override: override previously trained model
    :param class_weights: set weights for imbalanced datasets
    :return:
    """

    print 'Convolutional Neural Networks...'

    data = sequence.pad_sequences(data, maxlen=review_length, padding='post', truncating='post')

    x, x_test, y, y_test = train_test_split(data, labels, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size)

    print '  x_train shape: ', x_train.shape
    print '  x_val shape: ', x_val.shape
    print '  x_test shape: ', x_test.shape

    # Prepare embedding weights and convert inputs for specific model
    print '  Trained embeddings from corpus:', embeddings_from

    embedding_model, embedding_dim = get_embeddings(np.vstack((x_train, x_val)), vocabulary_inv)

    # Embedding initialization (random for non-present words in the corpus)
    if embedding_model is not None:
        embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                       else np.random.uniform(-0.25, 0.25, embedding_dim)
                                       for w in vocabulary_inv])]
    else:
        embedding_weights = None

    if not update_embeddings:
        x_train = embedding_weights[0][x_train]
        x_val = embedding_weights[0][x_val]
        x_test = embedding_weights[0][x_test]

        input_shape = (review_length, embedding_dim)
    else:
        input_shape = (review_length,)

    # Build or load CNN model
    model_name = cnn_model_path+'_%s_%s_%drev.h5' %('sent' if 'Negative' in classes else 'usef', embeddings_from, len(data))
    if os.path.exists(model_name) and not override:
        model = load_model(model_name)
        print 'CNN model "%s" loaded.' % os.path.split(model_name)[-1]

    else:
        print 'Building model...'
        model_input = Input(shape=input_shape)

        if not update_embeddings:
            z = Dropout(dropout_prob[0])(model_input)
        else:
            z = Embedding(input_dim=len(vocabulary_inv), output_dim=embedding_dim, name='embedding',
                          weights=embedding_weights, input_length=review_length)(model_input)
            z = Dropout(dropout_prob[0])(z)

        # Convolutional block
        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob[1])(z)
        z = Dense(hidden_dims, activation='relu')(z)
        model_output = Dense(1, activation='sigmoid', name='output_sent')(z)

        model = Model(model_input, model_output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Release memory
        del z, model_output, conv, conv_blocks

        # Train the model
        model.fit_generator(myGenerator(x_train, y_train, bs=batch_size), steps_per_epoch=len(x_train)/batch_size, epochs=num_epochs, callbacks=[early_stopping],
                  class_weight=class_weights, validation_data=(x_val, y_val), verbose=2)

        save_model(model, model_name)
        print 'CNN model "%s" saved.' % os.path.split(model_name)[-1]

    # Results display
    probabilities = model.predict(x_test, batch_size=batch_size)
    y_pred = [int(np.round(prob[0])) for prob in probabilities]

    print 'CNN test results:'
    print 'Accuracy: ', accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred, target_names=classes)
    #print confusion_matrix(y_test, y_pred, labels=[0,1])

    # Train results
    probabilities = model.predict(x_train, batch_size=batch_size)
    y_pred = [int(np.round(prob[0])) for prob in probabilities]

    print 'CNN train results:'
    print 'Accuracy: ', accuracy_score(y_train, y_pred)
    print classification_report(y_train, y_pred, target_names=classes)


def lstm_class(data, labels, vocabulary_inv, classes=['Negative', 'Positive'], override=True, class_weights=class_weights):
    """
    Build a LSTM model, train and show prediction results.
    :param data: list of indexed reviews
    :param labels: list of labels
    :param vocabulary_inv: dictionary matching {int:str}
    :param classes: classes names for printing results
    :param override: override previously trained model
    :param class_weights: set weights for imbalanced datasets
    :return:
    """

    print 'LSTM Recurrent Neural Networks...'

    data = sequence.pad_sequences(data, maxlen=review_length, padding='post', truncating='post')

    x, x_test, y, y_test = train_test_split(data, labels, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size)

    print '  x_train shape: ', x_train.shape
    print '  x_val shape: ', x_val.shape
    print '  x_test shape: ', x_test.shape

    # Prepare embedding weights and convert inputs for specific model
    print '  Trained embeddings from corpus:', embeddings_from

    embedding_model, embedding_dim = get_embeddings(np.vstack((x_train, x_val)), vocabulary_inv)

    # Embedding initialization (random for non-present words in the corpus)
    if embedding_model is not None:
        embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                       else np.random.uniform(-0.25, 0.25, embedding_dim)
                                       for w in vocabulary_inv])]
    else:
        embedding_weights = None

    if not update_embeddings:
        x_train = embedding_weights[0][x_train]
        x_val = embedding_weights[0][x_val]
        x_test = embedding_weights[0][x_test]

        input_shape = (review_length, embedding_dim)
    else:
        input_shape = (review_length,)

    # Build or load LSTM model
    model_name = lstm_model_path+'_%s_%s_%drev.h5' %('sent' if 'Negative' in classes else 'usef', embeddings_from, len(data))
    if os.path.exists(model_name) and not override:
        model = load_model(model_name)
        print 'LSTM model "%s" loaded.' % os.path.split(model_name)[-1]

    else:
        model_input = Input(shape=input_shape)

        if not update_embeddings:
            z = Dropout(dropout_prob[0])(model_input)
        else:
            z = Embedding(input_dim=len(vocabulary_inv), output_dim=embedding_dim, name='embedding',
                          weights=embedding_weights, input_length=review_length)(model_input)
            z = Dropout(dropout_prob[0])(z)

        # LSTM block
        # Forwards LSTM
        forwards = LSTM(hidden_dims)(z)
        # Backwards LSTM
        backwards = LSTM(hidden_dims, go_backwards=True)(z)
        z = Concatenate()([forwards, backwards])
        z = Dropout(dropout_prob[1])(z)

        z = Dense(hidden_dims, activation='relu')(z)
        model_output = Dense(1, activation='sigmoid')(z)

        model = Model(model_input, model_output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit_generator(myGenerator(x_train, y_train, bs=batch_size), steps_per_epoch=len(x_train) / batch_size,
                            epochs=num_epochs, callbacks=[early_stopping],
                            class_weight=class_weights, validation_data=(x_val, y_val), verbose=2)

        save_model(model, model_name)
        print 'LSTM model "%s" saved.' % os.path.split(model_name)[-1]

    # Results display
    probabilities = model.predict(x_test, batch_size=batch_size)
    y_pred = [int(np.round(prob[0])) for prob in probabilities]

    print 'LSTM test results:'
    print 'Accuracy: ', accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred, target_names=classes)


def svm_class(data, labels, vocabulary_inv, classes=['Negative', 'Positive']):
    """
    Build a SVM classifier, train and show prediction results.
    :param data: list of indexed reviews
    :param labels: list of labels
    :param vocabulary_inv: dictionary matching {int:str}
    :param classes: classes names for printing results
    :return:
    """

    print 'Support Vector Machines...'

    data = sequence.pad_sequences(data, maxlen=review_length, padding='post', truncating='post')

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

    print '  x_train shape: ', x_train.shape
    print '  x_test shape: ', x_test.shape

    # Prepare embedding weights and convert inputs for specific model
    print '  Trained embeddings from: ', 'reviews' if train_w2v else 'BoW'

    if train_w2v:
        embedding_model = train_word2vec(x_train, vocabulary_inv, emb_dim=embedding_dim_rev,
                                               min_word_count=min_word_count, context=context)
        embedding_dim = embedding_dim_rev

        if mean_embeddings:
            x_train = [[np.mean(embedding_model[word]) if word in embedding_model else np.random.uniform(-0.25, 0.25)
                        for word in rev] for rev in x_train]
            x_test = [[np.mean(embedding_model[word]) if word in embedding_model else np.random.uniform(-0.25, 0.25)
                       for word in rev] for rev in x_test]
        else:
            x_train = [np.hstack([embedding_model[word] if word in embedding_model else np.random.uniform(-0.25, 0.25, embedding_dim)
                        for word in rev]) for rev in x_train]
            x_test = [np.hstack([embedding_model[word] if word in embedding_model else np.random.uniform(-0.25, 0.25, embedding_dim)
                       for word in rev]) for rev in x_test]
        text_clf = svm.SVC()

    else:
        x_train = [' '.join([str(vocabulary_inv[x]) for x in review]) for review in x_train]
        x_test = [' '.join([str(vocabulary_inv[x]) for x in review]) for review in x_test]
        classifier = svm.SVC()
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', classifier), ])
    text_clf = text_clf.fit(x_train, y_train)
    y_pred = text_clf.predict(x_test)

    # Results display
    print 'SVM test results:'
    print 'Accuracy: ', accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred, target_names=classes)


def usef_result_from_sent(data, labels, vocabulary_inv, num_rev=248348, classes=['Not useful', 'Useful'], model_nm = 'cnn'):

    model_path = cnn_model_path if model_nm == 'cnn' else lstm_model_path
    model_name = model_path + '_sent_%s_%drev.h5' % (embeddings_from, num_rev)
    model = load_model(model_name)
    print 'Model "%s" loaded.' % os.path.split(model_name)[-1]

    data = sequence.pad_sequences(data, maxlen=review_length, padding='post', truncating='post')

    if not update_embeddings:
        embedding_model = train_word2vec(data, vocabulary_inv, emb_dim=embedding_dim_rev,
                                         min_word_count=min_word_count, context=context)
        embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                       else np.random.uniform(-0.25, 0.25, embedding_dim_rev)
                                       for w in vocabulary_inv])]
        data = embedding_weights[0][data]

    # Results display
    probabilities = model.predict(data, batch_size=batch_size)
    y_pred = [int(np.round(prob[0])) for prob in probabilities]

    print 'CNN test results:'
    print 'Accuracy: ', accuracy_score(labels, y_pred)
    print classification_report(labels, y_pred, target_names=classes)


def sent_weights_for_usef(data, labels, vocabulary_inv, num_rev=248348, model_nm = 'cnn'):

    # Load sent model and get output layer
    model_path = cnn_model_path if model_nm == 'cnn' else lstm_model_path
    model_name = model_path + '_sent_%s_%drev.h5' % (embeddings_from, num_rev)
    model = load_model(model_name)
    sent_layer = model.get_layer(name='output_sent')
    sent_layer.trainable = False
    print 'CNN output sent layer from model "%s" loaded.' % os.path.split(model_name)[-1]

    # Prepare data to classify
    data = sequence.pad_sequences(data, maxlen=review_length, padding='post', truncating='post')
    x, x_test, y, y_test = train_test_split(data, labels, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size)

    embedding_model, embedding_dim = get_embeddings(np.vstack((x_train, x_val)), vocabulary_inv)

    embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                       else np.random.uniform(-0.25, 0.25, embedding_dim)
                                       for w in vocabulary_inv])]
    del embedding_model

    if not update_embeddings:
        x_train = embedding_weights[0][x_train]
        x_val = embedding_weights[0][x_val]
        x_test = embedding_weights[0][x_test]

    print '  x_train shape: ', x_train.shape
    print '  x_val shape: ', x_val.shape
    print '  x_test shape: ', x_test.shape

    # Build CNN
    input_shape = (review_length, embedding_dim_rev) if not update_embeddings else (review_length,)
    model_input = Input(shape=input_shape)

    if not update_embeddings:
        z = Dropout(dropout_prob[0])(model_input)
    else:
        z = Embedding(input_dim=len(vocabulary_inv), output_dim=embedding_dim_rev, name='embedding',
                      weights=embedding_weights, input_length=review_length)(model_input)
        z = Dropout(dropout_prob[0])(z)

    # Sentiment layer
    z = Dense(hidden_dims, activation='relu')(z)
    z = Dropout(dropout_prob[0])(z)
    z = sent_layer(z)
    z = Dense(10, activation='relu')(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding='valid',
                             activation='relu',
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation='relu')(z)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit_generator(myGenerator(x_train, y_train, bs=batch_size), steps_per_epoch=len(x_train) / batch_size,
                        epochs=num_epochs, callbacks=[early_stopping],
                        class_weight=class_weights, validation_data=(x_val, y_val), verbose=2)

    # Results display
    probabilities = model.predict(x_test, batch_size=batch_size)
    y_pred = [int(np.round(prob[0])) for prob in probabilities]

    print 'CNN test results:'
    print 'Accuracy: ', accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred, target_names=['Not useful', 'Useful'])


# ------
def get_embeddings(words, vocabulary_inv):
    """
    Chooses embeddings among different corpus.
    :param words: sentences from reviews. Only needed to train word2vec.
    :param vocabulary_inv: dictionary mapping indexes to words from reviews. Only needed to train word2vec.
    :return embedding_model:
    :return embedding_dim: dimension of the vector representation of a word.
    """
    if embeddings_from == 'reviews':
        embedding_model = train_word2vec(words, vocabulary_inv, emb_dim=embedding_dim_rev,
                                           min_word_count=min_word_count, context=context)
        embedding_dim = embedding_dim_rev

    elif embeddings_from == 'web':

        def load_glove_model(glove_file):
            f = open(glove_file, 'r')
            model = {}
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = [float(val) for val in split_line[1:]]
                model[word] = embedding
            yield model, len(model[word])

        print 'Loading corpus...'
        embedding_model, embedding_dim = load_glove_model(web_path)

    elif embeddings_from == 'twitter':

        # Load Glove corpus
        print 'Loading corpus...'
        f = open(twit_path, 'r')
        embedding_model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(val) for val in split_line[1:]]
            embedding_model[word] = embedding

        embedding_dim = len(embedding_model[word])
        f.close()

    elif embeddings_from == 'news':
        news_embeddings = KeyedVectors.load_word2vec_format(news_path, binary=True)
        embedding_dim = len(news_embeddings['book'])
        print 'GoogleNews corpus loaded.'

        # Save memory reducing model dimension
        if update_embeddings:
            embedding_model = {}
            for word in vocabulary_inv:
                if word in news_embeddings.vocab:
                        embedding_model[word] = news_embeddings[word]
        else:
            embedding_model = news_embeddings
        print 'Number of words: ', len(embedding_model) if update_embeddings else len(embedding_model.vocab)

    elif embeddings_from == 'none':
        embedding_model = None
        embedding_dim = embedding_dim_rev
        global update_embeddings
        update_embeddings = True

    else:
        embedding_model = None
        embedding_dim = embedding_dim_rev
        global update_embeddings
        update_embeddings = True
        warnings.warn('Any embedding has been trained or loaded. Check variable: embeddings_from.', stacklevel=3)

    return embedding_model, embedding_dim


def train_word2vec(sentence_matrix, vocabulary_inv, emb_dim=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    - Inputs:
    sentence_matrix : int matrix: num_sentences x max_sentence_len
    vocabulary_inv  : dictionary matching {int:str}
    emb_dim         : Word vector (embedding) dimensionality
    min_word_count  : Minimum word count
    context         : Context window size
    """
    model_name = '{:s}_{:d}dim_{:d}minwords_{:d}context_{:d}words'.format(embeddings_from, emb_dim,
                                                                     min_word_count, context, len(vocabulary_inv))
    w2v_path = os.path.join(w2v_dir, model_name)

    if os.path.exists(w2v_path):
        embedding_model = word2vec.Word2Vec.load(w2v_path)
        print 'Loaded existing Word2Vec model "%s"' % os.path.split(w2v_path)[-1]

    else:
        # Initialize and train the model
        print 'Training Word2Vec model...'
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=emb_dim, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # Make the word2vec model much more memory-efficient
        embedding_model.init_sims(replace=True)

        # Save the model for later use
        if not os.path.exists(w2v_dir):
            os.mkdir(w2v_dir)
        print 'Saving Word2Vec model "%s"' % os.path.split(w2v_path)[-1]
        embedding_model.save(w2v_path)

    return embedding_model


def myGenerator(X_train, y_train, bs=batch_size):
    while 1:
        for i in range(len(X_train)/bs):
            yield X_train[i*bs:(i+1)*bs], y_train[i*bs:(i+1)*bs]