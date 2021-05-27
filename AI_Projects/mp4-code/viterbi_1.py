"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np
def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagfreq = {}        #Count occurences of tags
    tagpairfreq = {}    #Count occurences of tag sequence
    tagword = {}        #Count occurences of words for each tag
    transp = {}
    initp = {}
    emisp = {}
    laplace = 0.0001    #smoothing parameter
    vit = {}            #will store the max
    bit = {}            #pointer to previous tag
    output = []

#STEP 1 : Count occurrences of tags, tag pairs, tag/word pairs.
    for sentence in train:
        if sentence[1][1] not in initp: #store occurences of tag in start of a sentence
            initp[sentence[1][1]] = 1
        else:
            initp[sentence[1][1]] += 1
        for word in sentence:
            if word[1] not in tagfreq:  #store the occurences of each tag into a dict
                tagfreq[word[1]] = 1
            else:
                tagfreq[word[1]] += 1

            if word[1] not in tagword:  #store the occurences of each word for tag into a dict
                wordfreq = {}
                wordfreq[word[0]] = 1
                tagword[word[1]] = wordfreq

            else:
                if word[0] not in tagword[word[1]]:
                    tagword[word[1]][word[0]] = 1
                else:
                    tagword[word[1]][word[0]] += 1

        for i in range(len(sentence)-1):    #store the occurences of each tag sequence into a dict
            if (sentence[i][1], sentence[i+1][1]) not in tagpairfreq:
                tagpairfreq[(sentence[i][1], sentence[i+1][1])] = 1
            else:
                tagpairfreq[(sentence[i][1], sentence[i+1][1])] += 1

#STEP 2 : Compute smoothed probabilities
    for i in initp.keys():       #Precompute Initial Probabilities
        initp[i] = initp[i] / len(train)

    for tagb in tagfreq.keys():             #Precompute Transition probabilities
        for taga in tagfreq.keys():
            count = 0
            for j in tagpairfreq.keys():
                if j[0] == taga:
                    count += 1
            if (taga, tagb) not in tagpairfreq:
                transp[(taga,tagb)] = laplace / (tagfreq[tagb] + laplace*(count + 1))
            else:
                transp[(taga,tagb)] = (tagpairfreq[(taga, tagb)] + laplace) / (tagfreq[tagb] + laplace*(count + 1))

    for sentence in train:
        for word in sentence:
            if (word[0], word[1]) not in emisp:
                emisp[(word[0], word[1])] = (tagword[word[1]][word[0]] + laplace) / (tagfreq[word[1]] + laplace*(len(tagword[word[1]].keys())+ 1))

    for sentence in test:               #Compute probabilities and construct the trellis
        temp = []
        for i in range(1,len(sentence)):
            for tagb in tagfreq.keys():

                maxdict = {}            #store sum of log for each tag
                currword = sentence[i]
                if currword not in tagword[tagb]:   #Compute emission probability
                    emp = laplace / (tagfreq[tagb] + laplace*(len(tagword[tagb].keys())+ 1))
                else:
                    emp = emisp[(currword, tagb)]

                for taga in tagfreq.keys():
                    sum = 0

                    #STEP 3: Take the log of each probability
                    if i == 1:
                        if tagb in initp:
                            sum += np.log(initp[tagb])
                    if i != 1:
                        sum += vit[(i-1,taga)]
                    sum += np.log(transp[(taga,tagb)])
                    sum += np.log(emp)

                    maxdict[taga] = sum

                #STEP 4 : Construct the trellis.
                vit[(i,tagb)] = max(maxdict.values())
                bit[(i, tagb)] = max(maxdict, key=maxdict.get)

        #print(vit)
        #STEP 5: Return the best path through the trellis.
        n = len(sentence)-1
        maximum = -9999999999
        tagidx = None
        for tag in tagfreq.keys():
            if vit[(n,tag)] > maximum:
                maximum = vit[(n,tag)]
                tagidx = tag
        temp.append((sentence[n], tagidx))
        while n > 1:
            temp.append((sentence[n-1], bit[(n, tagidx)]))
            tagidx = bit[(n, tagidx)]
            n = n-1

        temp.append((sentence[0], 'START'))
        #print(sentence)
        #print(temp[::-1])
        output.append(temp[::-1])

    return output
