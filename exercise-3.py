import numpy as np
import os
from functions import *
from nltk.tag.perceptron import PerceptronTagger

#train sao os do conjunto de treino temario 2006
PATH_TRAIN_DOCS = './train/docs/'
PATH_TRAIN_RESUMES = './train/resumes/'
#test sao os docs do conjunto de teste (sao os mesmo docs do proj1)
PATH_TEST_DOCS = './SourceTextWithTitle/'
PATH_TEST_RESUMES = './AutoIdealExtractives/'

RESUME_LEN = 5
TRESHOLD = 0.15

docs = [f for f in os.listdir(PATH_TEST_DOCS)]
resumes = [f for f in os.listdir(PATH_TEST_RESUMES)]

trainDocs= [f for f in os.listdir(PATH_TRAIN_DOCS)]
trainResumes=[f for f in os.listdir(PATH_TEST_RESUMES)]

#
# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],
#
# ])
#
# y = np.array([-f)
#
# def perceptron_sgd(X, Y):
#     w = np.zeros(len(X[0]))
#     eta = 1
#     epochs = 20
#
#     for t in range(epochs):
#         for i, x in enumerate(X):
#             if (np.dot(X[i], w)*Y[i]) <= 0:
#                 w = w + eta*X[i]*Y[i]
#
#     return w


def main():
    global docs
    num_docs = len(docs)
    tagger1 = getTagger()
    chunker = getChunker()
    stopwords = getStopWords()
    trainTfIdf = dict()

    trainOriginalDocs=dict()
    trainInvertedListDoc=dict()
    trainDocSentenceTerm=dict()
    trainInvertedList=dict()
    trainNumTermsDoc=dict()
    trainNumTermsDocSentence=dict()


    #primeiro fazemos a secore de cada sentence contra o doc todo os docs de train
    setInvertedList(trainDocs, trainOriginalDocs, trainInvertedListDoc, trainDocSentenceTerm, trainInvertedList, tagger1, chunker, stopwords, trainNumTermsDoc, trainNumTermsDocSentence, PATH_TRAIN_DOCS)
    trainTfIdf = setTfIdf(trainDocSentenceTerm, trainInvertedList, trainOriginalDocs)
    #feature 1 sentence position
    trainSentencePos=dict()
    for doc in trainDocSentenceTerm:
        numPhrases=len(trainDocSentenceTerm[doc].keys())
        aux = [None for f in range(numPhrases)]
        for sentence in trainDocSentenceTerm[doc]:
            aux[sentence]= numPhrases/(sentence+1)
        trainSentencePos[doc]=aux

    #feature 2 Cosine similarity
    trainSentencesScores = getSentencesScoreDoc(trainDocs, trainDocSentenceTerm, trainInvertedList, trainOriginalDocs, trainInvertedListDoc)
    #print("tfidf", trainTfIdf)
    #print("setencesScore", trainSentencesScores)

    #precisamos de saber o indice das frases que pertencem ao documento
    trainSentencesInResume=dict()

    for doc in trainDocSentenceTerm:
        aux=[0 for i in range(len(trainDocSentenceTerm[doc].keys()))]
        resumeIndexes=getResumeIndexes(trainOriginalDocs[doc],doc,PATH_TRAIN_RESUMES)
        for index in resumeIndexes:
            aux[index]=1
        trainSentencesInResume[doc]=aux
        print("no doc "+doc+" as frases do resumo são os indices ",resumeIndexes)

    #contruir o array de features
    #for doc in trainDocSentenceTerm:
      #  trainSentencePos
      #  trainSentencesInResume
    #train
        
    #features dos doc de test

    #array de features do doc de test

    #test

    #get confiden on the classicication

    #use confidence to select top-5 sentences



main()