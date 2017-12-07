import numpy as np
import os
from functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
#train sao os do conjunto de treino temario 2006
PATH_TRAIN_DOCS = './train/docs/'
PATH_TRAIN_RESUMES = './train/resumes/'
#test sao os docs do conjunto de teste (sao os mesmo docs do proj1)

#PATH_TEST_DOCS = './SourceTextWithTitle/'
PATH_TEST_DOCS = './test/docs/'
#PATH_TEST_RESUMES = './AutoIdealExtractives/'
PATH_TEST_RESUMES = './test/resumes/'

resumes = [f for f in os.listdir(PATH_TEST_RESUMES)]

RESUME_LEN = 5
TRESHOLD = 0.15

docs = [f for f in os.listdir(PATH_TEST_DOCS)]
resumes = [f for f in os.listdir(PATH_TEST_RESUMES)]

trainDocs= [f for f in os.listdir(PATH_TRAIN_DOCS)]
trainResumes=[f for f in os.listdir(PATH_TEST_RESUMES)]

def SentencePosFeature(doc,docSentenceTerm):
    numPhrases=len(docSentenceTerm[doc].keys())
    aux = [None for f in range(numPhrases)]
    for sentence in docSentenceTerm[doc]:
        aux[sentence]= numPhrases/(sentence+1)
    return normalize(aux)

def SentenceInResume(doc,docSentenceTerm,OriginalDocs,path):
    aux=[0 for i in range(len(docSentenceTerm[doc].keys()))]
    resumeIndexes=getResumeIndexes(OriginalDocs[doc],doc,path)
    for index in resumeIndexes:
        aux[index]=1
    return aux

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

    # test
    tfIdf = dict()
    originalDocs = dict()
    invertedListDoc = dict()
    docSentenceTerm = dict()
    invertedList = dict()
    numTermsDoc = dict()
    numTermsDocSentence = dict()

    #primeiro fazemos a secore de cada sentence contra o doc todo os docs de train
    setInvertedList(trainDocs, trainOriginalDocs, trainInvertedListDoc, trainDocSentenceTerm, trainInvertedList, tagger1, chunker, stopwords, trainNumTermsDoc, trainNumTermsDocSentence, PATH_TRAIN_DOCS)
    trainTfIdf = setTfIdf(trainDocSentenceTerm, trainInvertedList, trainOriginalDocs)

    # primeiro fazemos a secore de cada sentence contra o doc todo os docs de train
    setInvertedList(docs, originalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker, stopwords,numTermsDoc, numTermsDocSentence, PATH_TEST_DOCS)
    tfIdf = setTfIdf(docSentenceTerm, invertedList, originalDocs)

    # contruir o array de features
    trainFeatures = []
    trainTarget = []
    # feature 2 Cosine similarity
    trainSentencesScores = getSentencesScoreDoc(trainDocs, trainDocSentenceTerm, trainInvertedList,trainOriginalDocs, trainInvertedListDoc)

    for doc in trainDocs:
        #feature 1 sentence position
        trainSentencePos=SentencePosFeature(doc,trainDocSentenceTerm)
        #precisamos de saber o indice das frases que pertencem ao documento
        trainSentencesInResume=SentenceInResume(doc,trainDocSentenceTerm,trainOriginalDocs,PATH_TRAIN_RESUMES)
        trainTarget += trainSentencesInResume
        scores=trainSentencesScores[doc]

        numPhrases = len(trainTfIdf[doc].keys())
        trainNumTermFrase=getFeatureNumTermsFrase(numPhrases, trainNumTermsDoc[doc], trainNumTermsDocSentence[doc])
        for i in range(len(trainSentencePos)):
            feature_sentence = []
            feature_sentence.append(trainSentencePos[i])
            feature_sentence.append(scores[i])
            feature_sentence.append(trainNumTermFrase[i])
            trainFeatures.append(feature_sentence)
        print(trainFeatures)

    sc = StandardScaler()
    sc.fit(trainFeatures)
    trainFeatures = sc.transform(trainFeatures)

    # array de features do doc de test
    #dicionario key =doc value  array[features sentence]
    testFeatures = dict()
    # feature 2 Cosine similarity
    sentencesScores = getSentencesScoreDoc(docs, docSentenceTerm, invertedList, originalDocs, invertedListDoc)

    for doc in docs:
        # feature 1 sentence position
        sentencePos = SentencePosFeature(doc, docSentenceTerm)
        scores = sentencesScores[doc]
        featureDoc=[]
        numPhrases = len(tfIdf[doc].keys())
        testNumTermFrase = normalize(getFeatureNumTermsFrase(numPhrases, numTermsDoc[doc], numTermsDocSentence[doc]))
        for i in range(len(sentencePos)):
            feature_sentence = []
            feature_sentence.append(sentencePos[i])
            feature_sentence.append(scores[i])
            feature_sentence.append(testNumTermFrase[i])
            featureDoc.append(feature_sentence)
        featureDoc = sc.transform(featureDoc)
        testFeatures[doc]=featureDoc
    print(testFeatures)
    ########################################################################################################
    # train
    ################

    random_state = 0
    n_iter = 40
    eta0 = 0.1
    ppn = Perceptron(n_iter=n_iter, eta0=eta0, random_state=random_state)
    ppn.fit(trainFeatures, trainTarget)
    #test
    print("predictions")
    mean_avg_precision1=0
    extracted = saveResumes(resumes, PATH_TEST_RESUMES)
    mean_avg_precision=0
    for doc in docs:
        #print("doc")
        test=sc.transform(testFeatures[doc])
        confidence=ppn.decision_function(test)
        print(confidence)
        resume = getOriginalSentence(doc, getFiveBestConfidences(confidence, RESUME_LEN), originalDocs)

        #for i in resume:
         #   print(i)
        #print(ppn.predict(testFeatures[doc]))
        mean_avg_precision += calc_avg_doc(resume, extracted[doc])
    #get confiden on the classicication
    mean_avg_precision = mean_avg_precision / num_docs
    print("MAP: " + str(mean_avg_precision))
    #use confidence to select top-5 sentences



main()