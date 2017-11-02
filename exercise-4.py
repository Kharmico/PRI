import os, nltk, string, itertools
import re
import math
import os
import operator
import nltk 
import nltk.data
import time
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

PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'

#Python understands the common character encoding used for Portuguese, ISO 8859-1 (ISO Latin 1).
ENCODING='iso8859-1/latin1'
grammar = "np: {(<adj>* <n>+ <prp>)? <adj>* <n>+}"	#utilizar este padrao, mas alterar consoante o utilizado para o portugues(?)
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
chunker = nltk.RegexpParser(grammar)

docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
resumes=[f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]

RESUME_LEN = 5
num_frases=0
sentInExtResumes = 0

invertedList = dict()#term-doc-sentence
invertedListDoc=dict()##doc-term-sentence
docSentenceTerm=dict()#doc-sentence-Term
avgSentenceLength=dict()#doc- value=avg sentencelength
OriginalDocs=dict()
num_frases_termo=dict() # key= termo, 
						#value = numero de fazes na colecao inteira em que o termo aparece

def simplify_tag(t):
		if "+" in t:
			return t[t.index("+")+1:]
		else:
			return t
testSents=floresta.tagged_sents()
testSents=[[(w.lower(), simplify_tag(t)) for (w,t) in sent] for sent in testSents if sent]
tagger0 = nltk.DefaultTagger('n')
tagger1 = nltk.UnigramTagger(testSents, backoff=tagger0)
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


#//////EXTRACT NOUN FRASES/////////////////////////////////////////////
#reference used to understand 
#https://gist.github.com/karimkhanp/4b7626a933759d0113d54b09acef24bf
#last vizited on 02/11/2017
def extract(unigrams):
	postoks = tagger1.tag(unigrams)
	tree = chunker.parse(postoks)
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
	unigrams= tokenizer.tokenize(text) # todas as palavras do texto 
	unigrams = [word.lower() for word in unigrams if word.lower() not in stopwords]
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
	global sent_tokenizer
	tokens = text.split('\n\n')
	frases_tokenize = []
	for t in tokens:
		frases_tokenize += sent_tokenizer.tokenize(t)
	frases_tokenize = [ sentence for sentence in frases_tokenize if (len(tokenizer.tokenize(sentence))!=0)]
	return frases_tokenize

def setInvertedList(docs):
	start=time.time()
	global invertedList,num_frases,OriginalDocs,invertedListDoc
	num_frases=0
	orderd = OrderedDict()
	for doc in docs:
		sum_sentence_length = 0
		f2 = open(PATH_SOURCE_TEXT + doc, "r")
		text = f2.read()
		f2.close()
		OriginalDocs[doc]=text
		sentences = DocToSentences(text)
		sentence_counter=1
		docSentenceTerm[doc] = dict()
		invertedListDoc[doc]=dict()
		for sentence in sentences:
			if len(sentence) !=0:
				aux_terms=stringToTerms(sentence)
				if(len(aux_terms)!=0):
					num_frases+=1
					aux_terms1=set(aux_terms)
					docSentenceTerm[doc][sentence_counter]=aux_terms
					sum_sentence_length+=len(aux_terms)
					for t in aux_terms:
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
		avgSentenceLength[doc]=(sum_sentence_length/sentence_counter)

def populateInvertedList(docs):
	global invertedList,OriginalDocs
	for doc in docs:
		text = OriginalDocs[doc]
		sentences = DocToSentences(text)
		sentence_counter=1
		for sentence in sentences:
			terms=stringToTerms(sentence)
			for t in terms:
				invertedList[t][doc][sentence_counter]=terms.count(t)
			sentence_counter+=1

	#////////////////////////////////
def idf(term,doc):
	global invertedList,OriginalDocs,num_frases,num_frases_termo
	ni=0
	n=0
	if term in num_frases_termo:
		ni=num_frases_termo[term]
	else:
		for doc in invertedList[term]:
			ni+=len(invertedList[term][doc].keys())
			num_frases_termo[term]=ni
	n=num_frases
	return math.log10((n-ni+0.5)/(ni+0.5))

def tf(term,doc,sentence,div):
	global invertedList,avgSentenceLength
	k1=1.2
	b=0.75
	numerador=invertedList[term][doc][sentence]*(k1+1)
	denominador=invertedList[term][doc][sentence]+k1*(1-b+(b*div))
	return (numerador/denominador)

######################OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO#########################

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

# This is for BM25
def setBM25():
	start=time.time()
	global invertedList, docSentenceTerm
	bm25 = dict()   # key = doc -sentence-term-peso termo
	for doc in docSentenceTerm:
		avgdl = avgSentenceLength[doc]
		bm25[doc] = dict()
		for sentence in docSentenceTerm[doc]:
			score = 0
			d = len(docSentenceTerm[doc][sentence])
			div = (d / avgdl)
			bm25[doc][sentence]=dict()
			for term in set(docSentenceTerm[doc][sentence]):
				idfAux = idf(term, doc)
				tfAux = tf(term, doc, sentence, div)
				bm25[doc][sentence][term]=tfAux * idfAux
	return bm25

def sqrtSomeSquares(doc, sentence, bm25):
	# TODO:
	value = 0
	aux = dict()
	aux = {k: v * v for k, v in bm25[doc][sentence].items()}
	value = sum(aux.values())
	return math.sqrt(value)


def calcpesoTermoDoc(doc, bm25):
	# TODO:
	start=time.time()
	pesosDoc = dict()
	maxTf = getFqMaxDoc(doc)
	for sentence in bm25[doc]:
		for term in bm25[doc][sentence]:
			if term not in pesosDoc.keys():
				pesosDoc[term] = ((getFqTermDoc(term, doc) / maxTf) * idf(term, doc))

	return pesosDoc


def sumMultiPesos(doc, sentence, pesosDoc, bm25):
	# TODO:
	value = 0
	maxTf = getFqMaxDoc(doc)  # Tf maximo dos termos no documento
	for term in bm25[doc][sentence]:
		value += (bm25[doc][sentence][term] * pesosDoc[term])
	# print("sumMultiPesos "+ str(sentence)+" "+term+" "+str(value))
	return value


def sqrtSomeSquaresDoc(pesosDoc):
	value = 0
	aux = dict()
	aux = {k: v * v for k, v in pesosDoc.items()}
	value = sum(aux.values())
	return math.sqrt(value)


def calculateScoreOfsentences(doc, bm25):
	# TODO:
	start=time.time()
	
	global invertedListDoc
	pesosDoc = dict()
	pesosDoc = calcpesoTermoDoc(doc, bm25)

	sentences_scores = dict()
	sqrt_some_squares_doc = sqrtSomeSquaresDoc(pesosDoc)
	for sentence in bm25[doc]:
		sqrt_some_squares = sqrtSomeSquares(doc, sentence, bm25)
		soma_mult_pesos = sumMultiPesos(doc, sentence, pesosDoc, bm25)
		#print("metricas "+str(sentence)+" doc "+str(doc))
		#print(str(sqrt_some_squares)+" "+str(soma_mult_pesos)+" "+str(sqrt_some_squares_doc))
		sentences_scores[sentence] = (soma_mult_pesos) / (sqrt_some_squares * sqrt_some_squares_doc)
	#print("time calculateScoreOfsentences: "+str((time.time()-start)))
	return sentences_scores

######################OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO#########################

def calculateScoreBM25(doc):
	#BM25 alg
	#TODO: calc peso do doc e das frases com a nova formla
	global docSentenceTerm
	avgdl=avgSentenceLength[doc]
	sentences_scores=dict()
	for sentence in docSentenceTerm[doc]:
		score=0
		d=len(docSentenceTerm[doc][sentence])
		div=(d/avgdl)
		for term in set(docSentenceTerm[doc][sentence]):
			idfAux=idf(term,doc)
			tfAux=tf(term,doc,sentence,div)
			#print("idf: " + str(idfAux) + "\ntf: " + str(tfAux))
			score+=tfAux*idfAux
		sentences_scores[sentence]=score
	return sentences_scores

def Soma_Mult_Pesos(doc,bm25,sentence1,sentence2):
	aux=0
	for term in bm25[doc][sentence1]:
		if(term in bm25[doc][sentence2]):
			aux+=bm25[doc][sentence1][term] * bm25[doc][sentence2][term]
	return aux

def Sqrt_Some_Squares(pesos):
	value=0
	for term in pesos:
		value+=pesos[term]*pesos[term]
	return value


def sumSim(doc,sentence,bm25,S):
	similaridades=[]
	sqrt_some_squares1=Sqrt_Some_Squares(bm25[doc][sentence])
	for s2 in S:
		sqrt_some_squares2=Sqrt_Some_Squares(bm25[doc][s2])
		soma_mult_pesos=Soma_Mult_Pesos(doc,bm25,sentence,s2)
		#print("metricas sumSim"+ str(sqrt_some_squares1)+ " "+str(sqrt_some_squares2)+" "+str(soma_mult_pesos))
		similaridades.append((soma_mult_pesos) / (sqrt_some_squares1 * sqrt_some_squares2))
	return sum(similaridades)
def getResume(doc,sentences_scores,bm25):
	#do ex4 here
	param = -1
	S=[]
	#calcular os trees melhores
	#sentences=list(bm25[doc].keys())
	for x in range(0,RESUME_LEN):
		_mmr=dict()
		for sentence in sentences_scores:
			_mmr[sentence]=(1-param)*sentences_scores[sentence]-(param*sumSim(doc,sentence,bm25,S))
		maxSent=max(_mmr.keys(), key=(lambda key: _mmr[key]))
		#print(_mmr)
		S.append(maxSent)
		del sentences_scores[maxSent]
	S.sort() #TODO
	return S
	#////////////////////////////////////

def getOriginalSentence(doc,idexs):
	global OriginalDocs
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = OriginalDocs[doc]
	sentences = DocToSentences(text)
	#print("doc: "+doc+ " tem "+str(len(sentences)))
	aux=[]
	for i in idexs:
	#   print("This is the value of i: " + str(i-1))
		aux.append(sentences[i-1])
	return aux

def resumeEx(docs):
	scoresDocs=dict()
	resumesDocs=dict()
	bm25 = setBM25()

	#print("resume bool")
	start=time.time()
	for doc in docs:
		scoresDocs[doc]=calculateScoreOfsentences(doc, bm25)
		resumesDocs[doc]=getOriginalSentence(doc,getResume(doc,scoresDocs[doc],bm25))
	#printResume(resumesDocs)
	#print("time resumeEx: "+str((time.time()-start)))

	return resumesDocs

# Save the Extracted resumes
def saveResumes():
	global resumes, sentInExtResumes
	extracted = dict()

	for doc in resumes:
		f2=open(PATH_AUTO_IDEAL_EXTRACTIVES + doc,'r')
		file = f2.read()
		f2.close()
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

def resumeTop5(docs):
	resumesDocs=dict()
	for doc in docs:
		aux=[]
		text=OriginalDocs[doc]
		sentences=DocToSentences(text)
		for i in range(0,5):
			aux.append(sentences[i])
		resumesDocs[doc]=aux
	return resumesDocs
def main():
	global docs
	resume=dict()
	setInvertedList(docs)
	resume1=resumeEx(docs)
	resume2=resumeTop5(docs)
	extracted = saveResumes()
	prec1 = 0
	rec1 = 0
	prec2=0
	rec2=0
	mean_avg_precision2=0
	mean_avg_precision1=0

	num_docs=len(docs)
	#print(num_docs)
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

	prec2=prec2/num_docs
	rec2=rec2/num_docs
	_f112=(2*rec2*prec2)/(rec2+prec2)
	mean_avg_precision2 = mean_avg_precision2/num_docs

	prec1=prec1/num_docs
	rec1=rec1/num_docs
	_f111=(2*rec2*prec2)/(rec2+prec2)
	mean_avg_precision1 = mean_avg_precision1/num_docs

	print("--- Metrics for MMR Approach")
	print("Precision: " + str(prec1))
	print("Recall : " + str(rec1))
	print("F1 : " + str(_f111))
	print("MAP : " + str(mean_avg_precision1))

	print("--- Metrics for Firsth 5 sentences Approach")
	print("Precision: " + str(prec2))
	print("Recall : " + str(rec2))
	print("F1 : " + str(_f112))
	print("MAP : " + str(mean_avg_precision2))

main()