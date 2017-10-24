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
invertedList = dict()
#docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
docs= ['smalltest.txt']
scores=dict() #key=doc value = sentence and respective score

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
	global terms,invertedList
	#para cada doc
	for doc in docs:
		#f2 = open(PATH_SOURCE_TEXT+doc, "r")
		print("doc "+doc)
		f2 = open(doc, "r")
		text = f2.read().lower()
		sentences = DocToSentences(text)
		
		for sentence in sentences:
			aux_terms=stringToTerms(sentence)
			for t in aux_terms:
				obj= TermClass()
				obj.term=t
				if t not in terms:
					terms[t]=obj
					invertedList[t]=dict()
				if doc not in invertedList[t]:
					invertedList[t][doc]=dict()
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
	print(str(invertedList))

	#////////////////////////////////
def maxTermfq(doc,sentence):
	#TODO:
	max=0
	for term in invertedList:
		if doc in invertedList[term]:
			if sentence in invertedList[term][doc]:
				if invertedList[term][doc][sentence] > max:
					max=invertedList[term][doc][sentence]
	return max

def idf(doc,term):
	#TODO:
	ni=len(invertedList[term][doc].keys())
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	f2 = open(doc, "r")
	text = f2.read().lower()
	sentences = DocToSentences(text)
	n=len(sentences)
	return math.log10(n/ni)

def setTfIdf(doc):
	#TODO:
	global invertedList
	tfIdf=dict()
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
	print(str(tfIdf))

def sqrtSomeSquares(doc,sentence):
	#TODO:
	return math.sqrt(value)


#def calcpesoTermoDoc(doc):
	#TODO:
	#pesosDoc[term]=((getFqTermDoc(term)/maxTf) * idf(term))
	#return pesosDoc

#def sumMultiPesos(sentence,pesosDoc):
	#TODO:

#def sqrtSomeSquaresDoc(pesosDoc):
	#value=0
	#TODO:
	#return math.sqrt(value)

#def calculateScoreOfsentences():
	#TODO:

#def getFqMaxDoc():
	#TODO:
	
#def getFqTermDoc(term):
	#TODO

#def getResume(doc):
	#TODO
	

#def printBest(tree_best):
	#TODO
	
	#////////////////////////////////////
def resumeEx1(docs):
	for doc in docs:
		setTfIdf(doc)

def main():
	global docs
	setInvertedList(docs)
	resumeEx1(docs)
	#resumeEx2(docs)
	
    


main()






