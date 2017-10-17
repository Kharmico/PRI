import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
terms=dict()
sentences =dict()
invertedList = dict()

class TermClass:
	term=""
	isf=None
	sf=None
	maxTf=None
	minTf=None
	def __eq__(self, other):
		if isinstance(other, SimilarityPair):
			return ((self.term == other.term))
		else:
			return False
	def __str__(self):
		return self.term

def stringToTerms(text):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(text) # todas as palavras do texto 

def stringToDictOfSentences(text):
	global terms,sentences,invertedList
	aux_sentences = sent_tokenize(text)
	i=1
	global sentences
	for line in aux_sentences:
		aux_terms=stringToTerms(line)
		sentences[i] = aux_terms
		for t in aux_terms:
			obj= TermClass()
			obj.term=t
			if t not in terms:
				terms[t]=obj
				invertedList[t]=dict()
		i+=1
		setNumOcorrencies()
		
		
def setNumOcorrencies():
	for t in terms:
			for s in sentences:
				fq=0
				fq=sentences[s].count(t)
				if fq >0 :
					if s not in invertedList[t]:
						invertedList[t][s]=fq

def maxTermfq(sentence):
	max=0
	for term in invertedList:
		if sentence in invertedList[term]:
			if invertedList[term][sentence] > value:
				max=value
	return max

def  teste(text):
	sentences = sent_tokenize(text)
	print(str(sentences))
	print("num sentences "+str(len(sentences)))
	vectorizer = TfidfVectorizer( use_idf=False)
	ft = vectorizer.fit_transform(sentences)
	print(ft)
	print(vectorizer.get_feature_names())


def readfile(filename):
	global terms,sentences
	f2=open(filename,"r")
	text=f2.read().lower()
	
	stringToDictOfSentences(text)
	teste(text)
	#for k in invertedList:
	#	print(str(k)+": "+str(invertedList[k]))

readfile("smalltest.txt")

