#!/usr/bin/python
# -*- coding: utf-8 -*-
import nltk
import string
from bs4 import BeautifulSoup
import traceback
from os import listdir
from os.path import isfile, join
import os
import re
from nltk.corpus import stopwords
from numpy import zeros
from math import log
import numpy
import operator
trainingIndex = ['14', '15', '16', '17']
testIndex = ['18']
month_acronym = 'feb'
DATASET = "E:\College\Spring15\AdvProkect\DownloadedDataset\\"
SANITIZED_DS = \
    "E:\College\Spring15\AdvProkect\DownloadedDataset\sanitized\\"


def extract_text(filehtml):
    soup = BeautifulSoup(filehtml)
    for elem in soup.findAll(['script', 'style']):
        elem.extract()
    text = soup.get_text()
    filetext = text.encode('utf-8')
    filetext = re.sub(r'[^\x00-\x7F]+', ' ', filetext)
    filetext = filetext.lower()

    # remove the punctuation

    no_punctuation = filetext.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [w for w in tokens if not w in stopwords.words('english'
                )]
    return '\n'.join(filtered)


# compute L2norm for the given vector

def L2norm(vector):
    sumSquare = 0.0
    for cell in vector:
        sumSquare = sumSquare + pow(cell, 2)
    denominator = pow(sumSquare, 0.5)
    if denominator==0:
      denominator=1
    # Now compute L2 norm finally

    length = len(vector)
    for i in range(length):
        vector[i] = float(vector[i]) / denominator
    return vector

#Find all whole words here:credits stackoverflow
def findWholeWord(word,text):
    return re.compile(r'\b({0})\b'.format(word), flags=re.IGNORECASE).findall(text)

# compute document-term matrix for given vector


def combined_vocabulary(trainingCorpus,testCorpus):
    vocab = []

    # Find the vocabulary in matrix here
    print 'Computing corpus for training data'
    for filetext in trainingCorpus:
        for word in filetext.split('\n'):
            if word not in vocab:
                vocab.append(word)
    print 'Computing corpus for training data'
    for filetext in testCorpus:
        for word in filetext.split('\n'):
            if word not in vocab:
                vocab.append(word)
    return vocab
def tf_matrix(corpus,vocab):
    

    
    row = len(corpus)
    column = len(vocab)
    docTermF = zeros((row, column))
    i = 0
    j = 0

    # Find simple document-term count

    for i in range(0, row):
        print 'In tf_matrix for doc='+str(i)
        maxi=-1
        for j in range(0, column):
            doc = corpus[i]
            word = vocab[j]
            
            search_res=findWholeWord(word,doc) 
            word_count=len(search_res)
            if word_count!=0:
                
                docTermF[i][j] = docTermF[i][j] + word_count
                if word_count>maxi:
                    maxi=word_count
            
        # Apply double normalization here
        for j in range(0, column):
            val=docTermF[i][j]
            if val==0:
                continue
            docTermF[i][j]=0.5+((0.5*val)/maxi)

    return docTermF


# compute idf matrix

def idf_matrix(corpus, vocab):
    docs = len(corpus)
    column = len(vocab)
    idf = zeros((column, column))
    i = 0
    j = 0

    # Find idf using inverse frequency

    for i in range(0, column):
        print 'idf for corpus='+str(i)
        #For smoothing keep count as 1
        count = 1
        word = vocab[i]
        for j in range(0, docs):
            doc = corpus[j]
            if len(findWholeWord(word,doc))!=0:
                count = count + 1
        val = 1+(float(docs) /  count)
        idf[i][i] = log(val)
    return idf


# Turn Test term frequency matrix corresponding to features in training because it
# be mapped against them only

def turnTestTfToTrain(docTermF,docTermTrainF, trainVocab, testVocab):
    docs = len(docTermF)
    column = len(trainVocab)
    tf = zeros((docs, column))
    for i in range(0, docs):
        doc = docTermF[i]
        for j in range(0, len(doc)):

            # Find the word in testVocab

            word = testVocab[j]
            count=-1
            index=0
            # Check if the corresponding word exists in trainVocab
            for words in trainVocab:
                if word==words:
                    count=1
                    break
                index=index+1
            if count==1:
                tf[i][index] = docTermF[i][j]
           
    return tf

# Compare training & testing docs

def compare(train, test,trainMapping,testMapping):
    testDocs = len(test)
    trainDocs = len(train)
    rankVal = {}
    for i in range(0, testDocs):
        print 'Compare for testDocs='+str(i)
        sums = 0.0
        testVec = test[i]
        for j in range(0, trainDocs):
            trainVec = train[j]
            cosineDoc=float(sum(testVec * trainVec))
            sums = sums + cosineDoc
            print 'cosine comparison of testDoc='+testMapping[str(i)]+' with train doc='+trainMapping[str(j)]+' is='+str(cosineDoc)
        rankVal[str(i)] = sums
    return rankValdef printRankedDocs(testMapping,rank):
    f=open('relevanceRank','w')
    count=1
    for (key,val) in rank:
        docName=testMapping[key]
        f.write(docName+' is ranked='+str(count)+' with cosine simi value='+str(val)+' \n')
    f.close()
#Map Evaluation parameter estimate for IR
def map_evaluation(testMapping,rank,totRelevantDocs):
    tot_count=0
    rele_count=0
    avg_prec=0.0
    for (key,val) in rank:
        docName=testMapping[key]
        tot_count=tot_count+1
        if 'relevant_' in docName:
            rele_count=rele_count+1
            avg_prec=avg_prec+(float(rele_count)/tot_count)
        if rele_count==totRelevantDocs:
            break
    map_val=avg_prec/totRelevantDocs
    return map_val
def testbase():
    f = None
    trainCount = 0
    testCount = 0
    totRelevantDocs=3
    trainCorpus = []
    testCorpus = []
    trainMapping = {}
    testMapping = {}
    for dirs in ['train', 'relevant', 'test']:
        onlyfiles = [fi for fi in listdir(dirs) if isfile(join(dirs, fi))]
        for files in onlyfiles:
            filehtml = open(join(dirs, files), 'r')
            try:
                filetext = extract_text(filehtml.read())
                nametag = dirs + '_' + files
             #   f = open(dirs + '\\' + nametag, 'w')
                print nametag

                #f.write(filetext)

                if dirs == 'train':
                    trainCorpus.append(filetext)
                    trainMapping[str(trainCount)] = nametag
                    trainCount = trainCount + 1
                elif dirs == 'test' or dirs == 'relevant':
                    testCorpus.append(filetext)
                    testMapping[str(testCount)] = nametag
                    testCount = testCount + 1
            except Exception:
                print traceback.format_exc()
            finally:
                if f:
                  f.close()
              
    #Compute vocabulary
    print 'Compute vocabulary for our dataset'
    vocab=combined_vocabulary( trainCorpus,testCorpus)   
    print 'Compute tf matrix for train set'
    docTermTrainF = tf_matrix(trainCorpus,vocab)
    print 'Compute idf matrix for train set'
    train_idf = idf_matrix(trainCorpus, vocab)
    print 'Compute tf matrix for test set'
    docTermTestF= tf_matrix(testCorpus,vocab)
    train_tfidf = numpy.dot(docTermTrainF, train_idf)
    test_tfidf = numpy.dot(docTermTestF, train_idf)
    #Compute L2 norm of both tf-idf vectors
    row_train=len(train_tfidf)
    for i in range(0, row_train):
        train_tfidf[i] = L2norm(train_tfidf[i])
    row_test=len(test_tfidf)
    for i in range(0, row_test):
        test_tfidf[i] = L2norm(test_tfidf[i])
    rank_val = compare(train_tfidf, test_tfidf,trainMapping,testMapping)
    sorted_docs = sorted(rank_val.items(), key=operator.itemgetter(1),reverse=True)
    printRankedDocs(testMapping,sorted_docs)
    map_val=map_evaluation(testMapping,sorted_docs,totRelevantDocs)
    print 'Map='+str(map_val)
if __name__ == '__main__':
    testbase()

            