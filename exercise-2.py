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
sentences = dict()  # chave e id da sentence valor e um array de termos
invertedList = dict()  # chave e' a palavra e o valor e' outro dicionario em qe a chave
# e' a sentence e o valor e' a frequencia (dicionario de dicionarios)
ts_tfIdf = dict()  # chave e termo, a segunda chave e sentence e o valor e o tf-idf do termo na frase
st_tfIdf = dict()  # chave e sentence, a segunda chave e termo e o valor
# e tf-idf do termo an frase os dois dicionarios sao iguais
original_sentences = dict()  # frazes originais.

PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'

#docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
docs= ['smalltest.txt']

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
    palavras_tokenize = tokenize.word_tokenize(text, language='portuguese')  # todas as palavras do texto
    return palavras_tokenize
def DocToSentences(text):
	frases_tokenize= tokenize.sent_tokenize(text, language='portuguese')
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
				invertedList[t][doc][sentence_counter]=sentence.count(t)
			sentence_counter+=1
	print(str(invertedList))

				


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

def main():
	global docs
	setInvertedList(docs)
    


main()






