import os, nltk, string, itertools
import re
import math
import os
import operator
import nltk
import nltk.data
import numpy as np
import time
import functions
from functions import DocToSentences, printBest, sqrtSomeSquares,  getResume, sumMultiPesos, setTfIdf, setInvertedList, stringToTerms
#nltk.download('punkt')
from collections import Counter
from collections import OrderedDict
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.chunk import RegexpParser
from nltk import ngrams
from nltk.corpus import floresta

#Python understands the common character encoding used for Portuguese, ISO 8859-1 (ISO Latin 1).
ENCODING='iso8859-1/latin1'
grammar = "np: {(<adj>* <n>+ <prp>)? <adj>* <n>+}"	#utilizar este padrao, mas alterar consoante o utilizado para o portugues(?)
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
chunker = nltk.RegexpParser(grammar)

PATH_TEXT = './teste/'
PATH_SOURCE_TEXT = './SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES = './ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES = './AutoIdealExtractives/'
RESUME_LEN = 5

sentInExtResumes = 0
terms = dict()
invertedList = dict()
invertedListDoc = dict()
docSentenceTerm = dict()
docs = [f for f in os.listdir(PATH_TEXT)]
OriginalDocs = dict()
resumes = [f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]
scores = dict()
scores2 = dict()
tfIdf = dict()
tf = dict()
num_frases_termo = dict()
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
tokenizer = RegexpTokenizer(r'\w+')





def createGraph(docs,tfIdf1):
    setofGraphs= dict()
    count = 0
    #print("tfidf", tfIdf1)
    for doc in docs:
        grafo = [[0 for x in range(len(tfIdf1[doc]))] for y in range(len(tfIdf1[doc]))]
        for sentence1 in tfIdf1[doc]:
            for sentence2 in tfIdf1[doc]:
                numerador = sumMultiPesos(doc,sentence1,sentence2,tfIdf1)
               # print("numerador", numerador)
                denominador1 = sqrtSomeSquares(doc,sentence1,tfIdf1)
               # print("denominador1", denominador1)
                denominador2 = sqrtSomeSquares(doc,sentence2,tfIdf1)
               # print("denominador2", denominador2)
                similarity = numerador / (denominador1*denominador2)
               # print("similarity", similarity)
                if similarity > 0.2 :
                    grafo[sentence1-1][sentence2-1]=similarity
        for sentence1 in tfIdf1[doc]:
            aux = " "
            for sentence2 in tfIdf1[doc]:
                aux+= " " + str(grafo[sentence1 - 1][sentence2 - 1])
            print(aux)
        setofGraphs[doc]=grafo
    return setofGraphs




def pageRank(numsentences, graph ):
    d = 0.15
    Po = 1/ numsentences

    #probpre = [Po for x in range(numsentences)]
   # probpos = [0 for x in range(numsentences)]
    probpre=dict()
    probpos =dict()
    for i in range(numsentences):
        probpre[i+1]=Po
        probpos[i+1]=0
   # for x in range(50):
    for i in range(numsentences):
        probpos[i+1] = (d / numsentences) + (1-d) * (somatorio(probpre, i, graph, numsentences))
    probpre = probpos
    return probpos

def somatorio(probpre, i, graph, numsentences):
    value = 0
    for j in range(numsentences) :
        counter= 0
        if graph[i][j] > 0:
            for k in range(numsentences):
                if graph[j][k] > 0:
                    counter +=1
            value =value+ (probpre[j+1] / counter)
    return value



def main():
    global docs
    resume1 = dict()
    setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList)
    tfIdf1 = setTfIdf(docSentenceTerm, invertedList, OriginalDocs)
    setofGraphs =  createGraph(docs,tfIdf1)
    for doc,graph in setofGraphs.items():
        numPhrases=len(tfIdf1[doc].keys())
        pagescore = pageRank(numPhrases, graph)
        resume1[doc]= getResume(pagescore, 5)
        printBest(doc, resume1, OriginalDocs)


main()






