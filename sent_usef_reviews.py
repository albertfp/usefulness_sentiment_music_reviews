import numpy as np
import os
from collections import Counter
import itertools
import pickle
import random

from text_clf import cnn_class, lstm_class, svm_class, usef_result_from_sent, sent_weights_for_usef
#import nltk
#import spacy

import warnings
#warnings.simplefilter('ignore')


# - Parameters -
num_albums = 1000000  # (max = 3564998)
my_path = './Datasets/'
fsample = my_path+'meta_CDs_and_Vinyl_novideo.json'
freviews = my_path+'reviews_CDs_and_Vinyl_novideo.json'
data_path = my_path+'data/reviews_%d' % num_albums

override = True
save_data = True
balance = True
#stop_words = nltk.corpus.stopwords.words('english')  # Only stopwords not punctuation (considered useful for sentiment)


def load_reviews():
    # TODO (DONE): save numpy data (np.save)
    """ Load and prepare data. """
    print 'Loading dataset...'

    reviews_list = []
    sentiment = []   # Whether if it is positive or negative
    usefulness = []  # Whether if it is helpful or not
    with open(freviews, 'r') as f:
        for line in f:
            data = eval(line)

            # Saving data to classify sentiment and usefulness
            reviews_list.append(data['reviewText'].lower())  # TODO: Interesting to add the 'Summary' section
            sentiment.append(int(data['overall'] > 3))       # Positive only if rating is 4 or 5 stars
            usefulness.append(data['helpful'])               # [a, b] = [positive votes, total votes] of review

            if len(sentiment) >= num_albums:
                break

    return reviews_list, sentiment, usefulness


def build_vocab(tokenized_texts):
    """ Build the vocabulary (word->index) and inverse vocabulary (index->word). """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*tokenized_texts))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def balance_data(labels):

    minor_label = 0 if sum(labels) > len(labels)/2 else 1
    minor_class = [[i, u] for i, u in enumerate(labels) if u == minor_label]
    class1 = [[i, u] for i, u in enumerate(labels) if u != minor_label][0:len(minor_class)]
    class1.extend(minor_class)
    random.shuffle(class1)
    new_labels = [u[1] for u in class1]
    new_idx = [u[0] for u in class1]

    return new_labels, new_idx


def evaluate_usefulness(useful):

    # Simple positive votes / total votes > 0.5
    reviews_idx = [i for i, _ in enumerate(useful) if useful[i][1] > 2]  # Tenemos en cuenta las reviews que tienen mas de dos votos
    usefulness = [int(float(useful[i][0]) / useful[i][1] > 0.5) for i in reviews_idx]

    return usefulness, reviews_idx


def main():

    if override or not os.path.exists(data_path + '_usef_labels.npy'):

        list_reviews, sent, useful = load_reviews()
        print '%d positive samples from %d.' % (sum(sent), len(sent))

        usefulness, usef_idx = evaluate_usefulness(useful)  # Only selected reviews (> 2 votes)
        print '%d useful reviews from %d voted/selected reviews (total reviews = %d).' % (sum(usefulness),
                                                                                          len(usef_idx),
                                                                                          len(useful))

        # Transfom reviews into organized lists of words
        print 'Tokenizing...'
        # NLTK:
        ## tokenized_reviews = [nltk.tokenize.word_tokenize(rev) for rev in list_reviews]
        # Spacy:
        ## nlp = spacy.load('en')
        ## print 'Spacy: English dictionary loaded.'
        ## tokenized_reviews = [nlp.tokenizer(unicode(rev)) for rev in list_reviews]
        # Split():
        tokenized_reviews = [rev.split() for rev in list_reviews]
        ## tokenized_reviews = [[word for word in rev if word not in stop_words] for rev in tokenized_reviews]

        print 'Building vocabulary...'
        voc, inv_voc = build_vocab(tokenized_reviews)
        pickle.dump(inv_voc, open(data_path + '_inv_voc.p', 'wb'))

        # Transform reviews into sequences of indexes
        print 'Preparing data to classify...'
        reviews = [[voc[word] for word in rev] for rev in tokenized_reviews]

        # Balance data
        if balance:
            sentiment, sent_idx = balance_data(sent)
            sent_reviews = np.array(reviews)[sent_idx]
            usefulness, usef_idx = balance_data(usefulness)
            usef_reviews = np.array(reviews)[usef_idx]

        else:
            sent_reviews = list_reviews
            sentiment = sent
            usef_reviews = np.array(reviews)[usef_idx]

        if save_data:
            # Save reviews for sentiment
            np.save(data_path + '_sent', sent_reviews)
            np.save(data_path + '_sent_labels', sentiment)
            # Save reviews for usefulness
            np.save(data_path + '_usef', usef_reviews)
            np.save(data_path + '_usef_labels', usefulness)

    else:
        print 'Loading data...'
        sent_reviews = np.load(data_path + '_sent.npy')
        sentiment = np.load(data_path + '_sent_labels.npy')
        usef_reviews = np.load(data_path + '_usef.npy')
        usefulness = np.load(data_path + '_usef_labels.npy')
        inv_voc = pickle.load(open(data_path + '_inv_voc.p', 'rb'))

    print '- Information -\n', 'Number of reviews used: ', num_albums
    print '%d useful reviews from %d voted/selected reviews.' % (sum(usefulness), len(usefulness))
    print '%d positive samples from %d.' % (sum(sentiment), len(sentiment))


    # Classification:

    # Usefulness
    print '\n- Classifying usefulness of reviews -'
    cnn_class(usef_reviews, usefulness, inv_voc, override=override, classes=['Not useful', 'Useful'])
    lstm_class(usef_reviews, usefulness, inv_voc, override=override, classes=['Not useful', 'Useful'])
    svm_class(usef_reviews, usefulness, inv_voc, classes=['Not useful', 'Useful'])

    # Sentiment
    print '\n- Classifying sentiment of reviews -'
    cnn_class(sent_reviews, sentiment, inv_voc, override=override)  # *Edit CNN parameters in text_clf.py script
    lstm_class(sent_reviews, sentiment, inv_voc, override=override)
    svm_class(sent_reviews, sentiment, inv_voc)


    # Usefulness using sentiment model (positive = useful / negative = non useful)
    print '\n- Classifying usefulness of reviews using sentiment model -'
    usef_result_from_sent(usef_reviews, usefulness, inv_voc, num_rev=len(sentiment))

    # Usefulness using sentiment layer (training a new classifier)
    print '\n- Classifying usefulness of reviews using sentiment layer -'
    sent_weights_for_usef(usef_reviews, usefulness, inv_voc, num_rev=len(sentiment))


if __name__ == '__main__':
    main()
