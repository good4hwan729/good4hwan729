# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    smoothing_parameter = 0.4
    dev_labels = []                         #return value
    pos_set = {}                            #dict for all positive train data
    neg_set = {}                            #dict for all negative train data
    totp = 0                                #n for positive
    totn = 0                                #n for negative

    stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    for j in range(len(train_set)):         #divide the train sets to class, and get count
        if train_labels[j] == 1:
            for m in train_set[j]:
                if m in pos_set:
                    pos_set[m] += 1
                else:
                    pos_set[m] = 1
                totp += 1
        else:
            for m in train_set[j]:
                if m in neg_set:
                    neg_set[m] += 1
                else:
                    neg_set[m] = 1
                totn += 1

    #sort_pos_set = sorted(pos_set.items(), key=lambda x:x[1], reverse=True)   #get top 5000 words for each set
    #sort_neg_set = sorted(neg_set.items(), key=lambda y:y[1], reverse=True)

    vp = len(pos_set.keys())                #getting number of unique words
    vn = len(neg_set.keys())

    for review in dev_set:                  #for each list of words
        finalnsum = 0                       #initialize the sum
        finalpsum = 0
        for w in review:                    #for each word in the list
            if w in stop_list:              #do not consider any words in the stop list
                continue
            if w not in pos_set.keys():            #check if it's an unknown word
                finalpsum += np.log(smoothing_parameter / (totp + smoothing_parameter*(vp + 1)))
            if w not in neg_set.keys():
                finalnsum += np.log(smoothing_parameter / (totn + smoothing_parameter*(vn + 1)))
            if w in pos_set.keys():
                finalpsum += np.log((pos_set[w] + smoothing_parameter) / (totp + smoothing_parameter*(vp + 1)))
            if w in neg_set.keys():
                finalnsum += np.log((neg_set[w] + smoothing_parameter) / (totn + smoothing_parameter*(vn + 1)))

        finalpsum += np.log(pos_prior)
        finalnsum += np.log(1-pos_prior)
        #print(abs(finalpsum))
        #print(abs(finalnsum))
        if finalpsum > finalnsum: #compare values
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    unigram_smoothing_parameter = 0.2
    bigram_smoothing_parameter = 0.1
    dev_labels = []                         #return value
    pos_set = {}                            #dict for all positive train data
    neg_set = {}                            #dict for all negative train data
    bi_pos_set = {}                            #dict for all positive train data
    bi_neg_set = {}                            #dict for all negative train data
    totp = 0                                #n for positive
    totn = 0                                #n for negative
    bitotp = 0                                #n for positive
    bitotn = 0                                #n for negative

    stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    for j in range(len(train_set)):         #divide the train sets to class, and get count
        if train_labels[j] == 1:
            for m in train_set[j]:
                if m in pos_set:
                    pos_set[m] += 1
                else:
                    pos_set[m] = 1
                totp += 1
        else:
            for m in train_set[j]:
                if m in neg_set:
                    neg_set[m] += 1
                else:
                    neg_set[m] = 1
                totn += 1

    for j in range(len(train_set)):         #divide the train sets to class, and get count
        if train_labels[j] == 1:
            for m in range(len(train_set[j])-1):
                if (train_set[j][m], train_set[j][m+1]) in bi_pos_set:
                    bi_pos_set[(train_set[j][m], train_set[j][m+1])] += 1
                else:
                    bi_pos_set[(train_set[j][m], train_set[j][m+1])] = 1
                bitotp += 1
        else:
            for m in range(len(train_set[j])-1):
                if (train_set[j][m], train_set[j][m+1]) in bi_neg_set:
                    bi_neg_set[(train_set[j][m], train_set[j][m+1])] += 1
                else:
                    bi_neg_set[(train_set[j][m], train_set[j][m+1])] = 1
                bitotn += 1

    vp = len(pos_set.keys())                #getting number of unique words
    vn = len(neg_set.keys())

    bivp = len(bi_pos_set.keys())                #getting number of unique words
    bivn = len(bi_neg_set.keys())

    for review in dev_set:                  #for each list of words
        finalnsum = 0                       #initialize the sum
        finalpsum = 0
        finalbinsum = 0                       #initialize the sum
        finalbipsum = 0
        for w in review:                    #for each word in the list
            if w in stop_list:              #do not consider any words in the stop list
                continue
            if w not in pos_set.keys():            #check if it's an unknown word
                finalpsum += np.log(unigram_smoothing_parameter / (totp + unigram_smoothing_parameter*(vp + 1)))
            if w not in neg_set.keys():
                finalnsum += np.log(unigram_smoothing_parameter / (totn + unigram_smoothing_parameter*(vn + 1)))
            if w in pos_set.keys():
                finalpsum += np.log((pos_set[w] + unigram_smoothing_parameter) / (totp + unigram_smoothing_parameter*(vp + 1)))
            if w in neg_set.keys():
                finalnsum += np.log((neg_set[w] + unigram_smoothing_parameter) / (totn + unigram_smoothing_parameter*(vn + 1)))

        finalpsum += np.log(pos_prior)
        finalnsum += np.log(1-pos_prior)

        for w in range(len(review)-1):                    #for each word in the list
            if review[w] or review[w+1] in stop_list:
                continue
            if (review[w], review[w+1]) not in bi_pos_set.keys():            #check if it's an unknown word
                finalbipsum += np.log(bigram_smoothing_parameter / (bitotp + bigram_smoothing_parameter*(bivp + 1)))
            if (review[w], review[w+1]) not in bi_neg_set.keys():
                finalbinsum += np.log(bigram_smoothing_parameter / (bitotn + bigram_smoothing_parameter*(bivn + 1)))
            if (review[w], review[w+1]) in bi_pos_set.keys():
                finalbipsum += np.log((bi_pos_set[(review[w], review[w+1])] + bigram_smoothing_parameter) / (bitotp + bigram_smoothing_parameter*(bivp + 1)))
            if (review[w], review[w+1]) in bi_neg_set.keys():
                finalbinsum += np.log((bi_neg_set[(review[w], review[w+1])] + bigram_smoothing_parameter) / (bitotn + bigram_smoothing_parameter*(bivn + 1)))

        finalbipsum += np.log(pos_prior)
        finalbinsum += np.log(1-pos_prior)

        compareP = (1 - bigram_lambda)*finalpsum + bigram_lambda*finalbipsum
        compareN = (1 - bigram_lambda)*finalnsum + bigram_lambda*finalbinsum

        if compareP > compareN:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels
