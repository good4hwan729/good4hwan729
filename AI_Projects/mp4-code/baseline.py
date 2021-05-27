"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    freqdct = {}
    output = []
    sentlist = []
    tagfreq = {}
    for sentence in train:
        for word in sentence:
            if word[1] not in tagfreq:
                tagfreq[word[1]] = 1
            else:
                tagfreq[word[1]] += 1
            if word[0] not in freqdct:
                tagdict = {}
                tagdict[word[1]] = 1
                freqdct[word[0]] = tagdict
            else:
                if word[1] not in freqdct[word[0]]:
                    freqdct[word[0]][word[1]] = 1
                else:
                    freqdct[word[0]][word[1]] += 1

    #sorted(freqdct.items(), key=lambda x: x[1])

    #print(freqdct)
    for sentence in test:
        sentlist = []
        for word in sentence:
            if word in freqdct:
                sentlist.append((word, max(freqdct[word], key=freqdct[word].get)))
            else:
                sentlist.append((word, max(tagfreq, key=tagfreq.get)))
        output.append(sentlist)
    #print(output)
    return output
