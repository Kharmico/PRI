import os, nltk, string, itertools
import re
import math
import os
import operator
import nltk
import nltk.data
import numpy as np
import time
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



def simplify_tag(t):
		if "+" in t:
			return t[t.index("+")+1:]
		else:
			return t

testSents=floresta.tagged_sents()
testSents=[[(w.lower(), simplify_tag(t)) for (w,t) in sent] for sent in testSents if sent]
tagger0 = nltk.DefaultTagger('n')
tagger0 = nltk.DefaultTagger('n')
tagger1 = nltk.UnigramTagger(testSents, backoff=tagger0)
tokenizer = RegexpTokenizer(r'\w+')

def extract(unigrams):
	postoks = tagger1.tag(unigrams)
	tree = chunker.parse(postoks)
	#print("This is the unigrams: " + str(unigrams))
	#print("---------------------------------------")
	def leaves(tree):
		"""Finds NP (nounphrase) leaf nodes of a chunk tree."""
		for subtree in tree.subtrees(filter = lambda t: t.label()=='np'):
			yield subtree.leaves()

	def acceptable_word(word):
		"""Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
		accepted = bool(2 <= len(word) <= 40
			and word.lower() not in stopwords)
		return accepted


	def get_terms(tree):
		for leaf in leaves(tree):
			term = [ w.lower() for w,t in leaf if acceptable_word(w) ]
			yield term

	terms = get_terms(tree)

	aux=[]
	for term in terms:
		t=''
		for word in term:
			t+= ' '+word
		t=t.strip()
		aux=aux+[t]
	return aux
#///////////////////////////////////////////////////////////////////////

def stringToTerms(text):
	#text=text.translate(None,string.punctuation)
	unigrams= tokenizer.tokenize(text) # todas as palavras do texto
	unigrams = [word.lower() for word in unigrams if word.lower() not in stopwords]
	#get noun phrases from text
	#if because of sentences conposed only by stop words 'nao so isso'
	noun_phrases=[]
	if(len(unigrams))!=0:
		noun_phrases=extract(unigrams)
	#2. we get the bigrams
	bigrams = ngrams(unigrams, 2)
	#3. we join the bigrams in a list like so (word word)
	text_bigrams = [' '.join(grams) for grams in bigrams]
	candidates = unigrams + text_bigrams +noun_phrases
	candidates=[word.lower() for word in set(candidates) if (word.lower() not in stopwords and 2 <= len(word))]
	return candidates# todas as palavras do texto




def DocToSentences(text):
    tokens = text.split('\n\n')
    frases_tokenize = []
    for t in tokens:
        frases_tokenize += sent_tokenizer.tokenize(t)
    frases_tokenize = [sentence for sentence in frases_tokenize if (len(tokenizer.tokenize(sentence)) != 0)]
    return frases_tokenize #retorna lista com as frases


def setInvertedList(docs):
    for doc in docs:
        f2 = open(PATH_TEXT + doc, "r")
        text = f2.read()
        text = text.lower()
        OriginalDocs[doc] = text
        sentences = DocToSentences(text)
        invertedListDoc[doc] = dict()
        docSentenceTerm[doc] = dict()
        sentence_counter = 1
        for sentence in sentences:
            if len(sentence) != 0:
                aux_terms = stringToTerms(sentence)
                if (len(aux_terms) != 0):
                    docSentenceTerm[doc][sentence_counter] = aux_terms
                    for t in aux_terms:
                        if t not in invertedListDoc[doc]:
                            invertedListDoc[doc][t] = dict()
                        if sentence_counter not in invertedListDoc[doc][t]:
                            invertedListDoc[doc][t][sentence_counter] = 0
                        invertedListDoc[doc][t][sentence_counter] += 1
                        if t not in invertedList:
                            invertedList[t] = dict()
                        if doc not in invertedList[t]:
                            invertedList[t][doc] = dict()
                        invertedList[t][doc][sentence_counter] = aux_terms.count(t)
                    sentence_counter += 1
        print(docSentenceTerm)# dic de dic onde key nome do doc value e' dic onde key e' numero da frase e value array dos termos da frase
        # print(invertedListDoc)# dicionario de dicionario onde key principal e' o nome do documento e seu value e' um dic com key = termo e value e' dic onde key  n.da frase e value n. vezes termo na frase
        # print(invertedList)#dic de dic onde key e' o termo e value e' um dic onde key e' o nome do doc e o value e' um dic onde a key e' o numero da frase e value e' o numero de ocorrencias do termo na frase



def maxTermfq(doc,sentence):
    max=0
    for term in invertedList:
        if doc in invertedList[term]:
            if sentence in invertedList[term][doc]:
                if invertedList[term][doc][sentence] > max:
                    max=invertedList[term][doc][sentence]
    return max


def getOriginalSentence(doc,idexs):
    global OriginalDocs
    text = OriginalDocs[doc]
    sentences = DocToSentences(text)
    aux=[]
    for i in idexs:
        aux.append(sentences[i-1])
        return aux

def idf(term,doc):
    num_frases = 0
    ni=0
    n=0
    text = OriginalDocs[doc].lower()
    sentences = DocToSentences(text)
    ni=len(invertedList[term][doc].keys())
    n=len(sentences)
    return math.log10(n/ni)

def setTfIdf():
    tfIdf=dict()
    for doc in docSentenceTerm:
        tfIdf[doc]=dict()
        for sentence in docSentenceTerm[doc]:
            tfIdf[doc][sentence]=dict()
            maxi=maxTermfq(doc,sentence)
            for term in set(docSentenceTerm[doc][sentence]):
                _idf=idf(term,doc)
                tf=invertedList[term][doc][sentence]
                value=(tf/maxi)*_idf
                tfIdf[doc][sentence][term]=value
    return tfIdf


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




def sqrtSomeSquaresDoc(pesosDoc):
    value=0
    aux=dict()
    aux={k: v*v for k, v in pesosDoc.items()}
    value=sum(aux.values())
    return math.sqrt(value)


def sqrtSomeSquares(doc,sentence,tfIdf):
    value=0
    aux=dict()
    aux={k: v*v for k, v in tfIdf[doc][sentence].items()}
    value=sum(aux.values())
    return math.sqrt(value)

def sumMultiPesos(doc,sentence1, sentence2, tfIdf):
    value=0
    aux = set(tfIdf[doc][sentence1].keys()).intersection(tfIdf[doc][sentence2].keys())
    print(sentence1, tfIdf[doc][sentence1])
    print(sentence2, tfIdf[doc][sentence2])
    print("auuuuuuux", aux)
    for term in aux:
        value += tfIdf[doc][sentence1][term] * tfIdf[doc][sentence2][term]
    return value


def getFqMaxDoc(doc):
    global invertedListDoc
    value=0
    for term in invertedListDoc[doc]:
        aux=0
        for sentence in invertedListDoc[doc][term]:
            aux+= invertedListDoc[doc][term][sentence]
        if aux > value:
            value=aux
    return value


def getFqTermDoc(term,doc):
    value=0
    global invertedListDoc
    if term in invertedListDoc[doc]:
        for sentence in invertedListDoc[doc][term]:
            value+= invertedListDoc[doc][term][sentence]
    return value

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

def getResume(sentences_scores):
	tree_best=[]
	#calcular os trees melhores
	for x in range(0,RESUME_LEN):
		maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
		tree_best.append(maxSent)
		del sentences_scores[maxSent]
	tree_best.sort()
	return tree_best

def printBest(doc, fivebest):
    sentences =DocToSentences(OriginalDocs[doc])
    for i in fivebest[doc]:
        print(sentences[int(i)-1])

def main():
    global docs
    resume1 = dict()
    setInvertedList(docs)
    tfIdf1 = setTfIdf()
    setofGraphs =  createGraph(docs,tfIdf1)
    for doc,graph in setofGraphs.items():
        numPhrases=len(tfIdf1[doc].keys())
        pagescore = pageRank(numPhrases, graph)
        #getResume(pagescore)
        resume1[doc]= getResume(pagescore)
        printBest(doc, resume1)


main()






