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
RESUME_LEN = 3

sentInExtResumes = 0
terms = dict()
invertedList = dict()#term-doc-sentence
invertedListDoc=dict()#doc-term-sentence
docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
resumes=[f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]
#docs= ['smalltest.txt']
scores=dict() #key=doc-sentence - respective score
#tfIdf=dict() #doc-sentence-term
				#dada a extrutura, do 
				#dicionario as vezes este e usado para iterar as frases de um texto
scores2=dict() #key=doc-sentence - respective score
tfIdf2=dict() #doc-sentence-term
				#dada a extrutura, do 
				#dicionario as vezes este e usado para iterar as frases de um texto
num_frases=0
num_frases_termo=dict() # key= termo, 
						#value = numero de fazes na colecao inteira em que o termo aparece

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
	global terms,invertedList,invertedListDoc,num_frases
	#para cada doc
	num_frases=0
	for doc in docs:
		f2 = open(PATH_SOURCE_TEXT+doc, "r")
		#f2 = open(doc, "r")
		text = f2.read().lower()
		sentences = DocToSentences(text)
		invertedListDoc[doc]=dict()
		sentence_counter=1
		for sentence in sentences:
			num_frases+=1
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
	#print("inverted list DOC:----------------------------------")
	#print(str(invertedListDoc))
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
	#print("inverted list:")
	#print(str(invertedList))

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
#true e ex1 false e idf do ex2
def idf(term,doc,bool):
	global num_frases,num_frases_termo,invertedList
	ni=0
	n=0
	f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = f2.read().lower()
	sentences = DocToSentences(text)
	if(bool):
		ni=len(invertedList[term][doc].keys())
		n=len(sentences)
	else:
		counter=0
		if term in num_frases_termo:
			ni=num_frases_termo[term]
		else:
			for doc in invertedList[term]:
				ni+=len(invertedList[term][doc].keys())
		n=num_frases
	return math.log10(n/ni)

def setTfIdf(bool):
	global invertedList
	tfIdf=dict()
	for term in invertedList:
		for doc in invertedList[term]:
			_idf=idf(term,doc,bool)
			for sentence in invertedList[term][doc]:
				maxi=maxTermfq(doc,sentence)
				tf=invertedList[term][doc][sentence]
				value=(tf/maxi)*_idf
				
				#inicializar os dicionarios
				if doc not in tfIdf:
					tfIdf[doc]=dict()
				if sentence not in tfIdf[doc]:
					tfIdf[doc][sentence]=dict()
				if term not in tfIdf[doc][sentence]:
					tfIdf[doc][sentence][term]=dict()

				tfIdf[doc][sentence][term]=value
	return tfIdf

def sqrtSomeSquares(doc,sentence,tfIdf):
	#TODO:
	value=0
	aux=dict()
	aux={k: v*v for k, v in tfIdf[doc][sentence].items()}
	value=sum(aux.values())
	return math.sqrt(value)


def calcpesoTermoDoc(doc,tfIdf,bool):
	#TODO:
	pesosDoc= dict()
	maxTf=getFqMaxDoc(doc)
	for sentence in tfIdf[doc]:
		for term in tfIdf[doc][sentence]:
			if term not in pesosDoc.keys():
				pesosDoc[term]=((getFqTermDoc(term,doc)/maxTf) * idf(term,doc,bool))
	return pesosDoc

def sumMultiPesos(doc,sentence,pesosDoc,tfIdf):
	#TODO:
	value=0
	maxTf=getFqMaxDoc(doc)# Tf maximo dos termos no documento
	for term in tfIdf[doc][sentence]:
		value+=(tfIdf[doc][sentence][term] * pesosDoc[term])
		#print("sumMultiPesos "+ str(sentence)+" "+term+" "+str(value))
	return value

def sqrtSomeSquaresDoc(pesosDoc):
	value=0
	aux=dict()
	aux={k: v*v for k, v in pesosDoc.items()}
	value=sum(aux.values())
	return math.sqrt(value)

def calculateScoreOfsentences(doc,tfIdf,bool):
	#TODO:
	global invertedListDoc
	pesosDoc=dict()
	pesosDoc=calcpesoTermoDoc(doc,tfIdf,bool)
	
	sentences_scores=dict()
	for sentence in tfIdf[doc]:
		sqrt_some_squares=sqrtSomeSquares(doc,sentence,tfIdf)
		soma_mult_pesos=sumMultiPesos(doc,sentence,pesosDoc,tfIdf)
		sqrt_some_squares_doc=sqrtSomeSquaresDoc(pesosDoc)
		#print("metricas "+str(sentence))
		#print(str(sqrt_some_squares)+" "+str(soma_mult_pesos)+" "+str(sqrt_some_squares_doc))
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
	for x in range(0,RESUME_LEN):
		maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
		tree_best.append(maxSent)
		del sentences_scores[maxSent]
	tree_best.sort()
	return tree_best
	#////////////////////////////////////

def printSentences(doc,idexs):
	f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = f2.read()
	sentences = DocToSentences(text)
	for i in idexs:
		print (sentences[i])

def printResume(resumesDocs):
	for doc in resumesDocs.keys():
		print("resume doc: "+doc)
		printSentences(doc,resumesDocs[doc])

def resumeEx(docs, bool):
	tfIdf1=dict()
	tfIdf1=setTfIdf(bool)
	scoresDocs=dict()
	resumesDocs=dict()
	print("resume bool")
	for doc in docs:
		scoresDocs[doc]=calculateScoreOfsentences(doc,tfIdf1,bool)
		resumesDocs[doc]=getResume(scoresDocs[doc])
	printResume(resumesDocs)
	return resumesDocs
	
# Save the Extracted resumes
def saveResumes():
	global resumes, sentInExtResumes
	extracted = dict()

	for doc in resumes:
		file = open(PATH_AUTO_IDEAL_EXTRACTIVES + doc).read()
		sentences = DocToSentences(file)
		docSeparate = doc.split("-")
		docToSave = docSeparate[1] + "-" + docSeparate[2]
		#print("Doc To Save: " + docToSave)
		extracted[docToSave] = sentences
		sentInExtResumes += len(sentences)
	return extracted

# Length (nº of sentences) of our own set
def calcA():
	global docs
	return (len(docs)*RESUME_LEN)

def intersectCalc(resume, extracted):
	counter = 0

	for doc in resume:
		print(doc)
		for sent in extracted[doc]:
			print(sent)
			print(resume[doc])
			if sent in resume[doc]:
				counter += 1
		#counter += len(set(resume[doc]).intersection(extracted[doc]))
	return counter

def precision(intersection, a):
	print("Intersection: " + str(intersection) + " and A: " + str(a))
	return (intersection/a)

def recall(intersection):
	global sentInExtResumes
	print("Intersection: " + str(intersection) + " and sentInExtResumes: " + str(sentInExtResumes))
	return (intersection/sentInExtResumes)

def main():
	global docs
	resume1=dict()
	resume2=dict()
	a=0
	setInvertedList(docs)
	resume1=resumeEx(docs, True)
	resume2=resumeEx(docs, False)
	a=calcA()
	extracted = saveResumes()
	intersection = intersectCalc(resume1, extracted)
	prec = precision(intersection, a)
	rec = recall(intersection)
	print("Precision: " + str(prec))
	print("Recall : " + str(rec))


main()






