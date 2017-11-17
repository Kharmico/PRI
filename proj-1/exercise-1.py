import math
import operator
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer

sentences = dict()  # chave e id da sentence valor e um array de termos
invertedList = dict()  # chave e' a palavra e o valor e' outro dicionario em qe a chave
# e' a sentence e o valor e' a frequencia (dicionario de dicionarios)
st_tfIdf = dict()  # chave e sentence, a segunda chave e termo e o valor
# e tf-idf do termo an frase os dois dicionarios sao iguais
original_sentences = dict()  # frazes originais.
FILE="smalltest.txt"

def stringToTerms(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return  tokenizer.tokenize(text) # divide a frase num array de termos

def stringToDictOfSentences(text):
    aux_sentences = tokenize.sent_tokenize(text, language='english') #texto em ingles
    i = 1
    for line in aux_sentences:
        aux_terms = stringToTerms(line)
        sentences[i] = aux_terms #sentences: dict() em qe a key e o n. da frase e o valor e um array dos termos da frase
        original_sentences[i] = line # original_sentences: dict() em qe a key n. da frase e o valor e a frase
        for t in aux_terms:
            if t not in invertedList:
                invertedList[t] = dict() # invertedList dict() keys sao os termos
        i += 1
        setNumOcorrencies()

def setNumOcorrencies():
    for t in invertedList:
        for s in sentences:
            fq = sentences[s].count(t)
            if fq > 0:
                if s not in invertedList[t]:
                    invertedList[t][s] = fq   # conta o numero de ocorrencias dos termos nas frases

def maxTermfq(sentence):
    max = 0
    for term in invertedList:
        if sentence in invertedList[term]:
            if invertedList[term][sentence] > max:
                max = invertedList[term][sentence]
    return max     #retorna o numero max de ocorrencias do termo qe ocorre mais vezes na frase


def idf(term):
    ni = len(invertedList[term].keys())#numero de frases em qe o termo i aparace
    n = len(sentences) #numero de frases
    return math.log10(n / ni) # calculo idf


def setTfIdf():
    for term in invertedList:
        for sentence in invertedList[term]:
            maxi = maxTermfq(sentence)
            tf = invertedList[term][sentence]
            if sentence not in st_tfIdf:
                st_tfIdf[sentence] = dict()
            if term not in st_tfIdf:
                st_tfIdf[sentence][term] = dict()
            value = (tf / maxi) * idf(term)
            st_tfIdf[sentence][term] = value   # tfidf = f*idf


def sqrtSomeSquares(sentence):
    aux = {k: v * v for k, v in st_tfIdf[sentence].items()} # quadrado
    value = sum(aux.values())#soma dos quadrados
    return math.sqrt(value) #raiz


def getFqMaxDoc():
    value = 0
    for term in invertedList:
        aux = 0
        for sentence in invertedList[term]:
            aux += invertedList[term][sentence]
        if aux > value:
            value = aux
    return value   #termo qe ocorre mais vezes no documento




def getFqTermDoc(term):
    value = 0
    if term in invertedList:
        for sentence in invertedList[term]:
            value += invertedList[term][sentence]
    return value   # retorna o numero de ocorrencias do termo no documento



def calcpesoTermoDoc():
    pesosDoc = dict()
    maxTf = getFqMaxDoc()
    for term in invertedList:
        pesosDoc[term] = ((getFqTermDoc(term) / maxTf) * idf(term))
    return pesosDoc #dic() em qe a key e o termo e o valor e tfidf


def sumMultiPesos(sentence, pesosDoc):
    value = 0
    for term in st_tfIdf[sentence]:
        value += (st_tfIdf[sentence][term] * pesosDoc[term])
    return value # numerador funcao sim


def sqrtSomeSquaresDoc(pesosDoc):
    aux = {k: v * v for k, v in pesosDoc.items()}
    value = sum(aux.values())
    return math.sqrt(value)


def calculateScoreOfsentences():
    pesosDoc = calcpesoTermoDoc()
    sentences_scores = dict()
    for sentence in sentences:
        sqrt_some_squares = sqrtSomeSquares(sentence)
        soma_mult_pesos = sumMultiPesos(sentence, pesosDoc)
        sqrt_some_squares_doc = sqrtSomeSquaresDoc(pesosDoc)
        sentences_scores[sentence] = (soma_mult_pesos) / (sqrt_some_squares * sqrt_some_squares_doc)
    return sentences_scores


# def getResume():
#     sentences_scores = calculateScoreOfsentences()
#     three_best = dict(sorted(sentences_scores.items(), key=operator.itemgetter(1), reverse=True)[:3])
#     return three_best

def getResume():
    sentences_scores=dict()
    sentences_scores=calculateScoreOfsentences()
    three_best=[]
    for x in range(0,3):
        maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
        three_best.append(maxSent)
        del sentences_scores[maxSent]
        three_best.sort()
    return three_best

def printBest(three_best):
    for i in three_best:
        print(original_sentences[i])


def readfile(filename):
    f2 = open(filename, "r")
    text = f2.read().lower()
    stringToDictOfSentences(text)
    setTfIdf()
    printBest(getResume())


readfile(FILE)
