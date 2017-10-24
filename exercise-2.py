import re
import math
import os
import operator
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import RegexpTokenizer


terms = dict()
invertedList = dict()#term-doc-sentence
invertedListDoc=dict()#doc-term-sentence
#docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
docs= ['smalltest.txt']
scores=dict() #key=doc value = sentence and respective score
tfIdf=dict() #doc-sentence-term

PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'



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
	return tokenizer.tokenize(text) # todas as palavras do texto 

def DocToSentences(text):
	frases_tokenize= tokenize.sent_tokenize(text, language='english')
	return frases_tokenize

def setInvertedList(docs):
	global terms,invertedList,invertedListDoc
	#para cada doc
	for doc in docs:
		#f2 = open(PATH_SOURCE_TEXT+doc, "r")
		f2 = open(doc, "r")
		text = f2.read().lower()
		sentences = DocToSentences(text)
		invertedListDoc[doc]=dict()
		sentence_counter=1
		for sentence in sentences:
			aux_terms=stringToTerms(sentence)
			for t in aux_terms:
				obj= TermClass()
				obj.term=t
				if t not in invertedListDoc[doc]:
					invertedListDoc[doc][t]=dict()
				if sentence_counter not in invertedListDoc[doc][t]:
					invertedListDoc[doc][t][sentence_counter]=0
				invertedListDoc[doc][t][sentence_counter]+=1
				if t not in terms:
					terms[t]=obj
					invertedList[t]=dict()
				if doc not in invertedList[t]:
					invertedList[t][doc]=dict()
			sentence_counter+=1
	print("inverted list DOC:----------------------------------")
	print(str(invertedListDoc))
	populateInvertedList(docs)

def populateInvertedList(docs):
	global invertedList
	for doc in docs:
		#f2 = open(PATH_SOURCE_TEXT+doc, "r")
		f2 = open(doc, "r")
		text = f2.read().lower()
		sentences = DocToSentences(text)
		sentence_counter=1		
		for sentence in sentences:
			terms=stringToTerms(sentence)
			for t in terms:
				invertedList[t][doc][sentence_counter]=terms.count(t)
			sentence_counter+=1
	print("inverted list:")
	print(str(invertedList))

	#////////////////////////////////
def maxTermfq(doc,sentence):
	global invertedListDoc
	max=0
	for term in invertedList:
		if doc in invertedList[term]:
			if sentence in invertedList[term][doc]:
				if invertedList[term][doc][sentence] > max:
					max=invertedList[term][doc][sentence]
	return max

def idf(doc,term):
	ni=len(invertedList[term][doc].keys())
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	f2 = open(doc, "r")
	text = f2.read().lower()
	sentences = DocToSentences(text)
	n=len(sentences)
	return math.log10(n/ni)

def setTfIdf():
	global invertedList,tfIdf
	for term in invertedList:
		for doc in invertedList[term]:
			for sentence in invertedList[term][doc]:
				maxi=maxTermfq(doc,sentence)
				tf=invertedList[term][doc][sentence]
				value=(tf/maxi)*idf(doc,term)
				
				#inicializar os dicionarios
				if doc not in tfIdf:
					tfIdf[doc]=dict()
				if sentence not in tfIdf[doc]:
					tfIdf[doc][sentence]=dict()
				if term not in tfIdf[doc][sentence]:
					tfIdf[doc][sentence][term]=dict()

				tfIdf[doc][sentence][term]=value

def sqrtSomeSquares(doc,sentence):
	#TODO:
	global tfIdf
	value=0
	aux=dict()
	aux={k: v*v for k, v in tfIdf[doc][sentence].items()}
	value=sum(aux.values())
	return math.sqrt(value)


def calcpesoTermoDoc(doc):
	#TODO:
	global tfIdf
	pesosDoc= dict()
	maxTf=getFqMaxDoc(doc)
	for term in tfIdf[doc][sentence]:
		pesosDoc[term]=((getFqTermDoc(term,doc)/maxTf) * idf(term,doc))
	return pesosDoc

def sumMultiPesos(doc,sentence,pesosDoc):
	#TODO:
	global tfIdf
	value=0
	maxTf=getFqMaxDoc(doc)# Tf maximo dos termos no documento
	for term in tfIdf[doc][sentence]:
		value+=(tfIdf[doc][sentence][term] * pesosDoc[term])
	return value

def sqrtSomeSquaresDoc(pesosDoc):
	value=0
	aux=dict()
	aux={k: v*v for k, v in pesosDoc.items()}
	value=sum(aux.values())
	return math.sqrt(value)

def calculateScoreOfsentences(doc):
	#TODO:
	pesosDoc=dict()
	pesosDoc=calcpesoTermoDoc(doc)
	
	sentences_scores[doc]=dict()
	for sentence in sentences:
		sqrt_some_squares=sqrtSomeSquares(sentence)
		soma_mult_pesos=sumMultiPesos(doc,sentence,pesosDoc)
		sqrt_some_squares_doc=sqrtSomeSquaresDoc(pesosDoc)
		sentences_scores[doc][sentence]=(soma_mult_pesos)/(sqrt_some_squares*sqrt_some_squares_doc)
	return sentences_scores[doc]
# fq do termo que se repete mais vezes no doc
def getFqMaxDoc(doc):
	global invertedListDoc
	value=0
	for term in invertedListDoc[doc]:
		aux=0
		for sentence in invertedListDoc[doc][term]:
			aux+= invertedList[doc][term][sentence]
		if aux > value:
			value=aux
	return value
#fq do termo que se repete mais vezes no doc
def getFqTermDoc(doc,term):
	value=0
	global invertedListDoc
	if term in invertedListDoc[doc]:
		for sentence in invertedListDoc[doc][term]:
			value+= invertedList[doc][term][sentence]
	return value	
	#////////////////////////////////////
def resumeEx1(docs):
	setTfIdf()
	scoresDocs=dict()
	for doc in docs:
		scoresDocs[doc]=calculateScoreOfsentences(doc)
	
def main():
	global docs
	setInvertedList(docs)
	resumeEx1(docs)
	#resumeEx2(docs)
	
    


main()






