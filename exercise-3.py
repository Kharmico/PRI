
import os, nltk, string, itertools
import re
import math
import os
import operator
import nltk 
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.chunk import RegexpParser
from nltk import ngrams
from nltk.corpus import floresta


 # #1. we obtain the unigrams
 #    unigrams = nltk.word_tokenize(text)
 #    #2. we get the bigrams
 #    bigrams = ngrams(unigrams, 2)
 #
 #
 # # POS Tagging each sentence
 #    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
 #
 #    grammar = "NP: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"
 #    chunker = nltk.chunk.regexp.RegexpParser(grammar)
 #    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
 #                                                        for tagged_sent in tagged_sents))
 #
 #    # This is the right way to go, bigrams and trigrams will not consider inter-phrase tokens
 #    candidates = [' '.join(word for word, pos, chunk in group).lower()
 #                      for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]

#////////////////////////////////////////////////////////////////////////

PATH_SOURCE_TEXT ='./SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES='./ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES='./AutoIdealExtractives/'
RESUME_LEN = 5

sentInExtResumes = 0
invertedList = dict()#term-doc-sentence
invertedListDoc=dict()#doc-term-sentence
docs = [f for f in os.listdir(PATH_SOURCE_TEXT)]
OriginalDocs=dict()
resumes=[f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]
#docs= ['smalltest.txt']
scores=dict() #key=doc-sentence - respective score
num_frases=0
num_frases_termo=dict() # key= termo, 
                        #value = numero de fazes na colecao inteira em que o termo aparece
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

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
	global stopwords
	text=text.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	unigrams= tokenizer.tokenize(text) # todas as palavras do texto 
	unigrams = [word for word in set(unigrams) if word not in stopwords]
	#2. we get the bigrams
	bigrams = ngrams(unigrams, 2)
	#3. we join the bigrams in a list like so (word word)
	text_bigrams = [' '.join(grams) for grams in bigrams]
	#print("text_bigrams")
	#i=0
	#for gram in text_bigrams:
	#	print(gram)#4. we join the unigram and bigram list to get all the candidates
	#	i+=1
	#print("n text_bigrams "+str(i))
	candidates = unigrams + text_bigrams
	
	return candidates# todas as palavras do texto 

def DocToSentences(text):
	global sent_tokenizer
	frases_tokenize = sent_tokenizer.tokenize(text)
	return frases_tokenize

def setInvertedList(docs):
    global invertedList,invertedListDoc,num_frases,OriginalDocs
    #para cada doc
    #print("inverted list")
    num_frases=0
    for doc in docs:
        f2 = open(PATH_SOURCE_TEXT+doc, "r")
        #f2 = open(doc, "r")
        text = f2.read()
        f2.close()
        OriginalDocs[doc]=text
        #text = text.lower()
        sentences = DocToSentences(text)
        invertedListDoc[doc]=dict()
        sentence_counter=1
        for sentence in sentences:
            num_frases+=1
            aux_terms=stringToTerms(sentence)
            for t in set(aux_terms):
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
    #print("inverted list:----------------------------------")
    #print(str(invertedList.keys()))
    #populateInvertedList(docs)

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
def idf(term,doc):
    global invertedList,OriginalDocs
    ni=0
    n=0
    text = OriginalDocs[doc].lower()
    sentences = DocToSentences(text)
    n=len(sentences)
    ni=len(invertedList[term][doc].keys())
    return math.log10(n/ni)

def setTfIdf():
    global invertedList
    tfIdf=dict()
    for term in invertedList:
        for doc in invertedList[term]:
            _idf=idf(term,doc)
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


def calcpesoTermoDoc(doc,tfIdf):
    pesosDoc= dict()
    maxTf=getFqMaxDoc(doc)
    for sentence in tfIdf[doc]:
        for term in tfIdf[doc][sentence]:
            if term not in pesosDoc:
                pesosDoc[term]=((getFqTermDoc(term,doc)/maxTf) * idf(term,doc))
    return pesosDoc

def sumMultiPesos(doc,sentence,pesosDoc,tfIdf):
    value=0
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

def calculateScoreOfsentences(doc,tfIdf):
    global invertedListDoc
    pesosDoc=dict()
    pesosDoc=calcpesoTermoDoc(doc,tfIdf)  
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
    tfIdf1=dict()
    tfIdf1=setTfIdf()
    scoresDocs=dict()
    resumesDocs=dict()
    #print("resume bool")
    for doc in docs:
        scoresDocs[doc]=calculateScoreOfsentences(doc,tfIdf1)
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






