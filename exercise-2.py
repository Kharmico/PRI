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


PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'

terms = dict()
invertedList = dict()#term-doc-sentence
invertedListDoc=dict()#doc-term-sentence
docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
#docs= ['smalltest.txt']
scores=dict() #key=doc-sentence - respective score
tfIdf=dict() #doc-sentence-term
				#dada a extrutura, do 
				#dicionario as vezes este e usado para iterar as frases de um texto




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
		f2 = open(PATH_SOURCE_TEXT+doc, "r")
		#f2 = open(doc, "r")
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
		f2 = open(PATH_SOURCE_TEXT+doc, "r")
		#f2 = open(doc, "r")
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

def idf(term,doc):
	ni=len(invertedList[term][doc].keys())
	f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
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
				value=(tf/maxi)*idf(term,doc)
				
				#inicializar os dicionarios
				if doc not in tfIdf:
					tfIdf[doc]=dict()
				if sentence not in tfIdf[doc]:
					tfIdf[doc][sentence]=dict()
				if term not in tfIdf[doc][sentence]:
					tfIdf[doc][sentence][term]=dict()

				tfIdf[doc][sentence][term]=value
	print(str(tfIdf))

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
	for sentence in tfIdf[doc]:
		for term in tfIdf[doc][sentence]:
			if term not in pesosDoc.keys():
				pesosDoc[term]=((getFqTermDoc(term,doc)/maxTf) * idf(term,doc))
	return pesosDoc

def sumMultiPesos(doc,sentence,pesosDoc):
	#TODO:
	global tfIdf
	value=0
	maxTf=getFqMaxDoc(doc)# Tf maximo dos termos no documento
	for term in tfIdf[doc][sentence]:
		value+=(tfIdf[doc][sentence][term] * pesosDoc[term])
		print("sumMultiPesos "+ str(sentence)+" "+term+" "+str(value))
	return value

def sqrtSomeSquaresDoc(pesosDoc):
	value=0
	aux=dict()
	aux={k: v*v for k, v in pesosDoc.items()}
	value=sum(aux.values())
	return math.sqrt(value)

def calculateScoreOfsentences(doc):
	#TODO:
	global invertedListDoc,tfIdf
	pesosDoc=dict()
	pesosDoc=calcpesoTermoDoc(doc)
	
	sentences_scores=dict()
	for sentence in tfIdf[doc]:
		sqrt_some_squares=sqrtSomeSquares(doc,sentence)
		soma_mult_pesos=sumMultiPesos(doc,sentence,pesosDoc)
		sqrt_some_squares_doc=sqrtSomeSquaresDoc(pesosDoc)
		print("metricas "+str(sentence))
		print(str(sqrt_some_squares)+" "+str(soma_mult_pesos)+" "+str(sqrt_some_squares_doc))
		sentences_scores[sentence]=(soma_mult_pesos)/(sqrt_some_squares*sqrt_some_squares_doc)
	return sentences_scores
# fq do termo que se repete mais vezes no doc
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
#fq do termo que se repete mais vezes no doc
def getFqTermDoc(term,doc):
	value=0
	global invertedListDoc
	if term in invertedListDoc[doc]:
		for sentence in invertedListDoc[doc][term]:
			value+= invertedListDoc[doc][term][sentence]
	return value	

def getResume(sentences_scores):
	tree_best=[]
	#calcular os trees melhores
	for x in range(0,5):
		maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
		tree_best.append(maxSent)
		del sentences_scores[maxSent]
	tree_best.sort()
	return tree_best
	#////////////////////////////////////
def printSentences(doc,idexs):
	f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = f2.read().lower()
	sentences = DocToSentences(text)
	for i in idexs:
		print (sentences[i])

def printResume(resumesDocs):
	for doc in resumesDocs.keys():
		print("resume doc: "+doc)
		printSentences(doc,resumesDocs[doc])
		print()

def resumeEx1(docs):
	setTfIdf()
	scoresDocs=dict()
	resumesDocs=dict()
	for doc in docs:
		scoresDocs[doc]=calculateScoreOfsentences(doc)
		resumesDocs[doc]=getResume(scoresDocs[doc])
	printResume(resumesDocs)
	
def main():
	global docs
	setInvertedList(docs)
	resumeEx1(docs)
	#resumeEx2(docs)
	
    


main()






