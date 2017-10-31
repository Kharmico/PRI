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
import time


PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'
RESUME_LEN = 5

sentInExtResumes = 0
terms = dict()
invertedList = dict()#term-doc-sentence
invertedListDoc=dict()#doc-term-sentence
docSentenceTerm=dict()
docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
OriginalDocs=dict()
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
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
tokenizer = RegexpTokenizer(r'\w+')
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
	return tokenizer.tokenize(text) # todas as palavras do texto 

def DocToSentences(text):
	global sent_tokenizer
#problema de paragrafos que ano terminao com ponto final
	tokens = text.split('\n\n')
	frases_tokenize = []
	for t in tokens:
		frases_tokenize += sent_tokenizer.tokenize(t)
	frases_tokenize = [ sentence for sentence in frases_tokenize if (len(tokenizer.tokenize(sentence))!=0)]
	return frases_tokenize


def setInvertedList(docs):
	global terms,invertedList,invertedListDoc,num_frases,OriginalDocs
	#para cada doc
	#start=time.time()
	num_frases=0
	for doc in docs:
		f2 = open(PATH_SOURCE_TEXT+doc, "r")
		#f2 = open(doc, "r")
		text = f2.read()
		OriginalDocs[doc]=text
		text = text.lower()
		sentences = DocToSentences(text)
		invertedListDoc[doc]=dict()
		docSentenceTerm[doc] = dict()
		sentence_counter=1
		for sentence in sentences:
			if len(sentence) !=0:
				aux_terms=stringToTerms(sentence)
				if(len(aux_terms)!=0):
					num_frases+=1
					aux_terms1=set(aux_terms)
					docSentenceTerm[doc][sentence_counter]=aux_terms
					for t in aux_terms:
						obj= TermClass()
						obj.term=t
						if t not in invertedListDoc[doc]:
							invertedListDoc[doc][t]=dict()
						if sentence_counter not in invertedListDoc[doc][t]:
							invertedListDoc[doc][t][sentence_counter]=0
						invertedListDoc[doc][t][sentence_counter]+=1
						if t not in invertedList:
							invertedList[t]=dict()
						if doc not in invertedList[t]:
							invertedList[t][doc]=dict()
						invertedList[t][doc][sentence_counter]=aux_terms.count(t)
					sentence_counter+=1
	#print("time setInvertedList: "+str(time.time()-start))
	#print("inverted list:----------------------------------")
	#print("inverted list DOC:----------------------------------")
	#print(str(invertedListDoc))
	#print(str(len(invertedList.keys())))

	#populateInvertedList(docs)

def populateInvertedList(docs):
	global invertedList,OriginalDocs
	for doc in docs:
		#f2 = open(PATH_SOURCE_TEXT+doc, "r")
		#f2 = open(doc, "r")
		text = OriginalDocs[doc].lower()
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
	global num_frases,num_frases_termo,invertedList,OriginalDocs
	ni=0
	n=0
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	if(bool):
		text = OriginalDocs[doc].lower()
		sentences = DocToSentences(text)
		ni=len(invertedList[term][doc].keys())
		n=len(sentences)
	else:
		counter=0
		if term in num_frases_termo:
			ni=num_frases_termo[term]
		else:
			for doc in invertedList[term]:
				ni+=len(invertedList[term][doc].keys())
			num_frases_termo[term]=ni
		n=num_frases
	return math.log10(n/ni)

def setTfIdf(bool):
	#start=time.time()
	global invertedList,docSentenceTerm
	tfIdf=dict()
	for doc in docSentenceTerm:
		tfIdf[doc]=dict()
		for sentence in docSentenceTerm[doc]:
			tfIdf[doc][sentence]=dict()
			maxi=maxTermfq(doc,sentence)
			for term in set(docSentenceTerm[doc][sentence]):
				_idf=idf(term,doc,bool)
				tf=invertedList[term][doc][sentence]
				value=(tf/maxi)*_idf
				tfIdf[doc][sentence][term]=value
	#print(str(tfIdf))
	#print("time setTfIdf: "+str(time.time()-start))
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
	#start=time.time()
	pesosDoc= dict()
	maxTf=getFqMaxDoc(doc)
	for sentence in tfIdf[doc]:
		for term in tfIdf[doc][sentence]:
			#if term not in pesosDoc.keys():
			pesosDoc[term]=((getFqTermDoc(term,doc)/maxTf) * idf(term,doc,bool))
	#print("time calcpesoTermoDoc: "+str((time.time()-start)))
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
	#start=time.time()
	global invertedListDoc
	pesosDoc=dict()
	pesosDoc=calcpesoTermoDoc(doc,tfIdf,bool)
	
	sentences_scores=dict()
	sqrt_some_squares_doc=sqrtSomeSquaresDoc(pesosDoc)
	for sentence in tfIdf[doc]:
		sqrt_some_squares=sqrtSomeSquares(doc,sentence,tfIdf)
		soma_mult_pesos=sumMultiPesos(doc,sentence,pesosDoc,tfIdf)
		#print("metricas "+str(sentence))
		#print(str(sqrt_some_squares)+" "+str(soma_mult_pesos)+" "+str(sqrt_some_squares_doc))
		sentences_scores[sentence]=(soma_mult_pesos)/(sqrt_some_squares*sqrt_some_squares_doc)
	#print("time calculateScoreOfsentences bool "+str(bool) +" : "+str(time.time()-start))
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
#fq do termo no doc
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

#def printResume(resumesDocs):
#	for doc in resumesDocs.keys():
#		print("resume doc: "+doc)
#		for sentence in resumesDocs[doc]:
#			print(sentence)

def getOriginalSentence(doc,idexs):
	global OriginalDocs
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = OriginalDocs[doc]
	sentences = DocToSentences(text)
	#print("doc: "+doc+ " tem "+str(len(sentences)))
	aux=[]
	for i in idexs:
	#	print("This is the value of i: " + str(i-1))
		aux.append(sentences[i-1])
	return aux

def resumeEx(docs, bool):
	#start=time.time()
	tfIdf1=dict()
	tfIdf1=setTfIdf(bool)
	scoresDocs=dict()
	resumesDocs=dict()
	#print("resume bool")
	for doc in docs:
		scoresDocs[doc]=calculateScoreOfsentences(doc,tfIdf1,bool)
		resumesDocs[doc]=getOriginalSentence(doc,getResume(scoresDocs[doc]))
	#printResume(resumesDocs)
	#print("time resumeEx "+str(bool)+" : "+str(time.time()-start))
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

def calc_avg_doc(our_ResumeSents, ideal_ResumeSents):
	iteration = 1
	recall_i = 0
	num_relevant_sentences = 0
	avg_precision = 0

	for sentence in our_ResumeSents:
		if sentence in ideal_ResumeSents:
			recall_i = 1
		else:
			recall_i = 0
		if sentence in ideal_ResumeSents:
			num_relevant_sentences += 1
		precision_i = num_relevant_sentences/iteration
		avg_precision += (precision_i*recall_i)
		iteration += 1
	avg_precision=avg_precision/len(ideal_ResumeSents)
	return avg_precision

def calc_MAP(our_Resume, ideal_Resume):
	mean_avg_precision = 0
	for doc in docs:
		mean_avg_precision += calc_avg_doc(our_Resume[doc], ideal_Resume[doc])

	#mean_avg_precision = mean_avg_precision/len(our_Resume.keys())
	return mean_avg_precision

def main():
	global docs
	resume1=dict()
	resume2=dict()
	setInvertedList(docs)
	resume1=resumeEx(docs, True)
	resume2=resumeEx(docs, False)
	extracted = saveResumes()
	prec1 = 0
	rec1 = 0
	mean_avg_precision1=0
	prec2 = 0
	rec2 = 0
	mean_avg_precision2=0

	num_docs=len(docs)
	for doc in docs:
		intersection1=set(resume1[doc]).intersection(extracted[doc])
		prec1 += len(intersection1)/len(resume1[doc])
		rec1 += len(intersection1)/len(extracted[doc])
		mean_avg_precision1 += calc_avg_doc(resume1[doc], extracted[doc])
		#////////////////for 2////////////
		intersection2=set(resume2[doc]).intersection(extracted[doc])
		prec2 += len(intersection2)/len(resume2[doc])
		rec2 += len(intersection2)/len(extracted[doc])
		mean_avg_precision2 += calc_avg_doc(resume2[doc], extracted[doc])
		
	prec1=prec1/num_docs
	rec1=rec1/num_docs
	_f11=(2*rec1*prec1)/(rec1+prec1)
	mean_avg_precision1 = mean_avg_precision1/num_docs

	prec2=prec2/num_docs
	rec2=rec2/num_docs
	_f12=(2*rec2*prec2)/(rec2+prec2)
	mean_avg_precision2 = mean_avg_precision2/num_docs
	
	

	print("--- Metrics for 1st Exercise Approach")
	print("Precision: " + str(prec1))
	print("Recall : " + str(rec1))
	print("F1 : " + str(_f11))
	print("MAP : " + str(mean_avg_precision1))
	print("--- Metrics for 2nd Exercise Simple Approach")
	print("Precision: " + str(prec2))
	print("Recall : " + str(rec2))
	print("F1 : " + str(_f12))
	print("MAP : " + str(mean_avg_precision2))

main()






