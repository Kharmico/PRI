import re
import math
import os
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer


terms = dict()
sentences = dict()  # chave e id da sentence valor e um array de termos
invertedList = dict()  # chave e' a palavra e o valor e' outro dicionario em qe a chave
# e' a sentence e o valor e' a frequencia (dicionario de dicionarios)
ts_tfIdf = dict()  # chave e termo, a segunda chave e sentence e o valor e o tf-idf do termo na frase
st_tfIdf = dict()  # chave e sentence, a segunda chave e termo e o valor
# e tf-idf do termo an frase os dois dicionarios sao iguais
original_sentences = dict()  # frazes originais.

CAMINHO ='./textos/'

files = [f for f in os.listdir(CAMINHO)]
print(files)




class TermClass:
    term = ""
    isf = None
    sf = None
    maxTf = None
    minTf = None

    def __eq__(self, other):  # equivalente a equals
        if isinstance(other, TermClass):
            return ((self.term == other.term))
        else:
            return False

    def __str__(self):  # equivalente a to string
        return self.term


def stringToTerms(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)  # todas as palavras do texto


def stringToDictOfSentences(text):
    global terms, sentences, invertedList, original_sentences
    aux_sentences = sent_tokenize(text)
    i = 1
    global sentences
    for line in aux_sentences:
        aux_terms = stringToTerms(line)
        sentences[i] = aux_terms
        original_sentences[i] = line
        for t in aux_terms:
            obj = TermClass()
            obj.term = t
            if t not in terms:
                terms[t] = obj
                invertedList[t] = dict()
        i += 1
        setNumOcorrencies()


def setNumOcorrencies():
    for t in terms:
        for s in sentences:
            fq = 0
            fq = sentences[s].count(t)
            if fq > 0:
                if s not in invertedList[t]:
                    invertedList[t][s] = fq


def maxTermfq(sentence):
    max = 0
    for term in invertedList:
        if sentence in invertedList[term]:
            if invertedList[term][sentence] > max:
                max = invertedList[term][sentence]
    return max


def idf(term):
    global sentences
    ni = len(invertedList[term].keys())
    n = len(sentences.keys())
    return math.log10(n / ni)


def setTfIdf():
    global invertedList, ts_tfIdf, st_tfIdf
    for term in invertedList:

        for sentence in invertedList[term]:
            maxi = maxTermfq(sentence)
            tf = invertedList[term][sentence]
            if term not in ts_tfIdf:
                ts_tfIdf[term] = dict()
            if sentence not in st_tfIdf:
                st_tfIdf[sentence] = dict()
            if term not in st_tfIdf:
                st_tfIdf[sentence][term] = dict()

            ts_tfIdf[term][sentence] = dict()
            value = 0
            value = (tf / maxi) * idf(term)
            ts_tfIdf[term][sentence] = value
            st_tfIdf[sentence][term] = value


def sqrtSomeSquares(sentence):
    # untested
    value = 0
    aux = dict()
    aux = {k: v * v for k, v in st_tfIdf[sentence].items()}
    # print(str(aux))
    value = sum(aux.values())
    return math.sqrt(value)


def calcpesoTermoDoc():
    global terms
    pesosDoc = dict()
    maxTf = getFqMaxDoc()
    for term in terms:
        pesosDoc[term] = ((getFqTermDoc(term) / maxTf) * idf(term))
    return pesosDoc


def sumMultiPesos(sentence, pesosDoc):
    global st_tfIdf, sentences
    value = 0
    maxTf = getFqMaxDoc()  # Tf maximo dos termos no documento
    for term in sentences[sentence]:
        value += (st_tfIdf[sentence][term] * pesosDoc[term])
    return value


def sqrtSomeSquaresDoc(pesosDoc):
    value = 0
    aux = dict()
    aux = {k: v * v for k, v in pesosDoc.items()}
    value = sum(aux.values())
    return math.sqrt(value)


def calculateScoreOfsentences():
    global sentences
    pesosDoc = dict()
    pesosDoc = calcpesoTermoDoc()
    sentences_scores = dict()
    for sentence in sentences:
        sqrt_some_squares = sqrtSomeSquares(sentence)
        soma_mult_pesos = sumMultiPesos(sentence, pesosDoc)
        sqrt_some_squares_doc = sqrtSomeSquaresDoc(pesosDoc)
        sentences_scores[sentence] = (soma_mult_pesos) / (sqrt_some_squares * sqrt_some_squares_doc)
    print(sentences_scores)
    return sentences_scores


def getFqMaxDoc():
    value = 0
    global invertedList
    for term in invertedList:
        aux = 0
        for sentence in invertedList[term]:
            aux += invertedList[term][sentence]
        if aux > value:
            value = aux
    return value


def getFqTermDoc(term):
    value = 0
    global invertedList
    if term in invertedList:
        for sentence in invertedList[term]:
            value += invertedList[term][sentence]
    return value


def getResume():
    sentences_scores = dict()
    sentences_scores = calculateScoreOfsentences()
    tree_best = []
    # calcular os trees melhores
    for x in range(0, 3):
        maxSent = max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
        tree_best.append(maxSent)
        del sentences_scores[maxSent]
    tree_best.sort()
    return tree_best


def printBest(tree_best):
    global original_sentences
    for i in tree_best:
        print(original_sentences[i])


# #def readfile(filename):
#     global ts_tfIdf, terms, sentences
#     f2 = open(filename, "r")
#     text = f2.read().lower()
#     stringToDictOfSentences(text)
#     setTfIdf()
#     printBest(getResume())
#     print(sentences)
#     print(ts_tfIdf)

#readfile("smalltest.txt")

def readfile(files):
    f1 = files
    for i in f1:
        f2 = open(CAMINHO+i, "r")
        text = f2.read().lower()


readfile(files)






