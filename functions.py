import math
import nltk.data
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.corpus import floresta


PATH_TEXT = './teste/'
PATH_AUTO_IDEAL_EXTRACTIVES = './AutoIdealExtractives/'
RESUME_LEN = 5

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
tokenizer = RegexpTokenizer(r'\w+')
#Python understands the common character encoding used for Portuguese, ISO 8859-1 (ISO Latin 1).
ENCODING='iso8859-1/latin1'
grammar = "np: {(<adj>* <n>+ <prp>)? <adj>* <n>+}"	#utilizar este padrao, mas alterar consoante o utilizado para o portugues(?)
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
chunker = nltk.RegexpParser(grammar)

PATH_TEXT = './teste/'
PATH_SOURCE_TEXT = './SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES = './ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES = './AutoIdealExtractives/'
RESUME_LEN = 5




def simplify_tag(t):
		if "+" in t:
			return t[t.index("+")+1:]
		else:
			return t

testSents=floresta.tagged_sents()
testSents=[[(w.lower(), simplify_tag(t)) for (w,t) in sent] for sent in testSents if sent]
tagger0 = nltk.DefaultTagger('n')
tagger0 = nltk.DefaultTagger('n')
tagger1 = nltk.UnigramTagger(testSents, backoff=tagger0)
tokenizer = RegexpTokenizer(r'\w+')


def DocToSentences(text):
    tokens_tmp = text.split('\n\n')
    tokens=[]
    for t in tokens_tmp:
        tokens = tokens +text.split('\n')
    frases_tokenize = []
    for t in tokens:
        frases_tokenize += sent_tokenizer.tokenize(t)
    frases_tokenize = [sentence for sentence in frases_tokenize if (len(tokenizer.tokenize(sentence)) != 0)]
    return frases_tokenize #retorna lista com as frases

def printBest(doc, fivebest, OriginalDocs):
    sentences =DocToSentences(OriginalDocs[doc])
    for i in fivebest[doc]:
        print(sentences[int(i)-1])

def sqrtSomeSquaresDoc(pesosDoc):
    value=0
    aux=dict()
    aux={k: v*v for k, v in pesosDoc.items()}
    value=sum(aux.values())
    return math.sqrt(value)

def sqrtSomeSquares(doc,sentence,tfIdf):
    value=0
    aux=dict()
    aux={k: v*v for k, v in tfIdf[doc][sentence].items()}
    value=sum(aux.values())
    return math.sqrt(value)

def maxTermfq(doc,sentence, invertedList):
    max=0
    for term in invertedList:
        if doc in invertedList[term]:
            if sentence in invertedList[term][doc]:
                if invertedList[term][doc][sentence] > max:
                    max=invertedList[term][doc][sentence]
    return max

def getOriginalSentence(doc,idexs, OriginalDocs):
    text = OriginalDocs[doc]
    sentences = DocToSentences(text)
    aux=[]
    for i in idexs:
        aux.append(sentences[i-1])
        return aux


def stringToTerms(text):
	#text=text.translate(None,string.punctuation)
	unigrams= tokenizer.tokenize(text) # todas as palavras do texto
	unigrams = [word.lower() for word in unigrams if word.lower() not in stopwords]
	#get noun phrases from text
	#if because of sentences conposed only by stop words 'nao so isso'
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

def extract(unigrams):
	postoks = tagger1.tag(unigrams)
	tree = chunker.parse(postoks)
	#print("This is the unigrams: " + str(unigrams))
	#print("---------------------------------------")
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



def getFqMaxDoc(doc, invertedListDoc):
    value=0
    for term in invertedListDoc[doc]:
        aux=0
        for sentence in invertedListDoc[doc][term]:
            aux+= invertedListDoc[doc][term][sentence]
        if aux > value:
            value=aux
    return value


def getResume(sentences_scores, resumelen):
	tree_best=[]
	#calcular os trees melhores
	for x in range(0,resumelen):
		maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
		tree_best.append(maxSent)
		del sentences_scores[maxSent]
	tree_best.sort()
	return tree_best

def sumMultiPesos(doc,sentence1, sentence2, tfIdf):
    value=0
    aux = set(tfIdf[doc][sentence1].keys()).intersection(tfIdf[doc][sentence2].keys())
   # print(sentence1, tfIdf[doc][sentence1])
   # print(sentence2, tfIdf[doc][sentence2])
   # print("auuuuuuux", aux)
    for term in aux:
        value += tfIdf[doc][sentence1][term] * tfIdf[doc][sentence2][term]
    return value


def idf(term,doc, OriginalDocs, invertedList):
    num_frases = 0
    ni=0
    n=0
    text = OriginalDocs[doc].lower()
    sentences = DocToSentences(text)
    ni=len(invertedList[term][doc].keys())
    n=len(sentences)
    return math.log10(n/ni)

def setTfIdf(docSentenceTerm, invertedList, OriginalDocs):
    tfIdf=dict()
    for doc in docSentenceTerm:
        tfIdf[doc]=dict()
        for sentence in docSentenceTerm[doc]:
            tfIdf[doc][sentence]=dict()
            maxi=maxTermfq(doc,sentence, invertedList)
            for term in set(docSentenceTerm[doc][sentence]):
                _idf=idf(term,doc, OriginalDocs, invertedList)
                tf=invertedList[term][doc][sentence]
                value=(tf/maxi)*_idf
                tfIdf[doc][sentence][term]=value
    return tfIdf



def setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList):
    for doc in docs:
        f2 = open(PATH_TEXT + doc, "r")
        text = f2.read()
        text = text.lower()
        OriginalDocs[doc] = text
        sentences = DocToSentences(text)
        invertedListDoc[doc] = dict()
        docSentenceTerm[doc] = dict()
        sentence_counter = 1
        for sentence in sentences:
            if len(sentence) != 0:
                aux_terms = stringToTerms(sentence)
                if (len(aux_terms) != 0):
                    docSentenceTerm[doc][sentence_counter] = aux_terms
                    for t in aux_terms:
                        if t not in invertedListDoc[doc]:
                            invertedListDoc[doc][t] = dict()
                        if sentence_counter not in invertedListDoc[doc][t]:
                            invertedListDoc[doc][t][sentence_counter] = 0
                        invertedListDoc[doc][t][sentence_counter] += 1
                        if t not in invertedList:
                            invertedList[t] = dict()
                        if doc not in invertedList[t]:
                            invertedList[t][doc] = dict()
                        invertedList[t][doc][sentence_counter] = aux_terms.count(t)
                    sentence_counter += 1
        #print(docSentenceTerm)# dic de dic onde key nome do doc value e' dic onde key e' numero da frase e value array dos termos da frase
        # print(invertedListDoc)# dicionario de dicionario onde key principal e' o nome do documento e seu value e' um dic com key = termo e value e' dic onde key  n.da frase e value n. vezes termo na frase
        # print(invertedList)#dic de dic onde key e' o termo e value e' um dic onde key e' o nome do doc e o value e' um dic onde a key e' o numero da frase e value e' o numero de ocorrencias do termo na frase

def saveResumes(resumes):
    extracted = dict()
    sentInExtResumes = 0
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