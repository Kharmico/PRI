import os, nltk, string, itertools
import re
import math
import os
import operator
import nltk 
import nltk.data
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

PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'

#Python understands the common character encoding used for Portuguese, ISO 8859-1 (ISO Latin 1).
ENCODING='ISO 8859-1'
grammar = "NP: {(<ADJ>* <NOUN>+ <PRP>)? <ADJ>* <NOUN>+}"	#utilizar este padrao, mas alterar consoante o utilizado para o portugues(?)
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
chunker = nltk.RegexpParser(grammar)

docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
#docs= ['smalltest.txt']
resumes=[f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]

RESUME_LEN = 5
num_frases=0
sentInExtResumes = 0

invertedList = dict()#term-doc-sentence
docSentenceTerm=dict()#doc-sentence-Term
avgSentenceLength=dict()#doc- value=avg sentencelength
OriginalDocs=dict()
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


#//////EXTRACT NOUN FRASES/////////////////////////////////////////////
def extract(unigrams):

    def simplify_tag(t):
        if "+" in t:
            return t[t.index("+")+1:]
        else:
            return t

    testSents=floresta.tagged_sents()
    testSents=[[(w.lower(), simplify_tag(t)) for (w,t) in sent] for sent in testSents if sent]
    tagger0 = nltk.DefaultTagger('n')
    tagger1 = nltk.UnigramTagger(testSents, backoff=tagger0)
    postoks = tagger1.tag(unigrams)
	#postoks = nltk.tag.pos_tag(unigrams)
    tree = chunker.parse(postoks)
    #print("This is the unigrams: " + str(unigrams))
    #print("---------------------------------------")
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def normalise(word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = lemmatizer.lemmatize(word)
        return word

    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(2 <= len(word) <= 40
            and word.lower() not in stopwords)
        return accepted


    def get_terms(tree):
        for leaf in leaves(tree):
            term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
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

	tokenizer = RegexpTokenizer(r'\w+')
	unigrams= tokenizer.tokenize(text) # todas as palavras do texto 
	#get noun phrases from text
	noun_phrases=extract(unigrams)
	unigrams = [word.lower() for word in set(unigrams) if word.lower() not in stopwords]
	#2. we get the bigrams
	bigrams = ngrams(unigrams, 2)
	#3. we join the bigrams in a list like so (word word)
	text_bigrams = [' '.join(grams) for grams in bigrams]
	
	candidates = unigrams + text_bigrams +noun_phrases
	candidates=[word.lower() for word in candidates if (word.lower() not in stopwords and 2 <= len(word))]
	return candidates# todas as palavras do texto 

def DocToSentences(text):
	global sent_tokenizer
	#problema de paragrafos que nao terminam com ponto final
	#text = text.replace('\n\n', '.\n').replace('..\n', '.\n')
	tokens = text.split('\n\n')
	frases_tokenize = []
	for t in tokens:
		frases_tokenize += sent_tokenizer.tokenize(t)
	return frases_tokenize

def setInvertedList(docs):
    global invertedList,num_frases,OriginalDocs
    #para cada doc
    #print("inverted list")
    num_frases=0
    orderd = OrderedDict()
    for doc in docs:
    	sum_sentence_length = 0
    	f2 = open(PATH_SOURCE_TEXT + doc, "r")
    	#f2 = open(doc, "r")
    	text = f2.read()
    	f2.close()
    	OriginalDocs[doc]=text
    	#text = text.lower()
    	sentences = DocToSentences(text)
    	sentence_counter=1
    	docSentenceTerm[doc] = dict()
    	#print("SENTENCES: " + str(sentences))
    	for sentence in sentences:
    		num_frases+=1
    		aux_terms=stringToTerms(sentence)
    		aux_terms1=set(aux_terms)
    		docSentenceTerm[doc][sentence_counter]=aux_terms
    		sum_sentence_length+=len(aux_terms)
    		for t in aux_terms:
    			if t not in invertedList:
    				invertedList[t]=dict()
    			if doc not in invertedList[t]:
    				invertedList[t][doc]=dict()
    			#invertedList[t][doc][sentence_counter]=aux_terms.count(t)
    		sentence_counter+=1
    	avgSentenceLength[doc]=(sum_sentence_length/sentence_counter)
    #print("inverted list:----------------------------------")
    orderd=OrderedDict(sorted(invertedList.items()))
    for k, v in orderd.items():
    	print("This is a key: " + str(k))
    #print(str(invertedList.keys()))
    #print(str(len(invertedList.keys())))
    populateInvertedList(docs)

def populateInvertedList(docs):
    global invertedList,OriginalDocs
    for doc in docs:
        #f2 = open(PATH_SOURCE_TEXT+doc, "r")
        #f2 = open(doc, "r")
        text = OriginalDocs[doc]
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
    global invertedList, docSentenceTerm
    bm25 = dict()   # key = doc; value = dict of sentences with value of sentences_score
    for doc in docSentenceTerm:
        avgdl = avgSentenceLength[doc]
        sentences_scores = dict()
        for sentence in docSentenceTerm[doc]:
            score = 0
            d = len(docSentenceTerm[doc][sentence])
            div = (d / avgdl)
            for term in set(docSentenceTerm[doc][sentence]):
                idfAux = idf(term, doc)
                tfAux = tf(term, doc, sentence, div)
                # print("idf: " + str(idfAux) + "\ntf: " + str(tfAux))
                score += tfAux * idfAux
            sentences_scores[sentence] = score
        bm25[doc] = sentences_scores
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
    global invertedListDoc
    pesosDoc = dict()
    pesosDoc = calcpesoTermoDoc(doc, bm25)

    sentences_scores = dict()
    for sentence in bm25[doc]:
        sqrt_some_squares = sqrtSomeSquares(doc, sentence, bm25)
        soma_mult_pesos = sumMultiPesos(doc, sentence, pesosDoc, bm25)
        sqrt_some_squares_doc = sqrtSomeSquaresDoc(pesosDoc)
        # print("metricas "+str(sentence))
        # print(str(sqrt_some_squares)+" "+str(soma_mult_pesos)+" "+str(sqrt_some_squares_doc))
        sentences_scores[sentence] = (soma_mult_pesos) / (sqrt_some_squares * sqrt_some_squares_doc)
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

def getResume(sentences_scores):
    tree_best=[]
    #calcular os trees melhores
    for x in range(0,RESUME_LEN):
        maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
        tree_best.append(maxSent)
        del sentences_scores[maxSent]
    tree_best.sort() #TODO
    return tree_best
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
    for doc in docs:
        scoresDocs[doc]=calculateScoreOfsentences(doc, bm25)
        resumesDocs[doc]=getOriginalSentence(doc,getResume(scoresDocs[doc]))
    #printResume(resumesDocs)
    return resumesDocs
    
# Save the Extracted resumes
def saveResumes():
    global resumes, sentInExtResumes
    extracted = dict()

    for doc in resumes:
    	f2=open(PATH_AUTO_IDEAL_EXTRACTIVES + doc)
    	file = f2.read()
    	f2.close()
    	sentences = DocToSentences(file)
    	docSeparate = doc.split("-")
    	docToSave = docSeparate[1] + "-" + docSeparate[2]#print("Doc To Save: " + docToSave)
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


def main():
    global docs
    resume=dict()
    setInvertedList(docs)
    resume=resumeEx(docs)
    extracted = saveResumes()
    prec = 0
    rec = 0
    mean_avg_precision=0

    num_docs=len(docs)
    #print(num_docs)
    for doc in docs:
        intersection=set(resume[doc]).intersection(extracted[doc])
        prec += len(intersection)/len(resume[doc])
        rec += len(intersection)/len(extracted[doc])
        mean_avg_precision += calc_avg_doc(resume[doc], extracted[doc])
        #////////////////for 2////////////
        
    prec=prec/num_docs
    rec=rec/num_docs
    _f11=(2*rec*prec)/(rec+prec)
    mean_avg_precision = mean_avg_precision/num_docs

    print("--- Metrics for 1st Exercise Approach")
    print("Precision: " + str(prec))
    print("Recall : " + str(rec))
    print("F1 : " + str(_f11))
    print("MAP : " + str(mean_avg_precision))

main()