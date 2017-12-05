import math
import nltk.data
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.corpus import floresta


#Python understands the common character encoding used for Portuguese, ISO 8859-1 (ISO Latin 1).
ENCODING='iso8859-1/latin1'
TRESHOLD = 0.15


def getTagger():
    testSents = floresta.tagged_sents()
    testSents = [[(w.lower(), simplify_tag(t)) for (w, t) in sent] for sent in testSents if sent]
    tagger0 = nltk.DefaultTagger('n')
    tagger1 = nltk.UnigramTagger(testSents, backoff=tagger0)
    return tagger1

def getChunker():
    grammar = "np: {(<adj>* <n>+ <prp>)? <adj>* <n>+}"  # utilizar este padrao, mas alterar consoante o utilizado para o portugues(?)
    chunker = nltk.RegexpParser(grammar)
    return chunker

def  getStopWords():
    return set(nltk.corpus.stopwords.words('portuguese'))

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t


def DocToSentences(text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = text.split('\n\n')
    #tokens=[]
    #for t in tokens_tmp:
    #    tokens = tokens +text.split('\n')
    frases_tokenize = []
    for t in tokens:
        frases_tokenize += sent_tokenizer.tokenize(t)
    frases_tokenize = [sentence for sentence in frases_tokenize if (len(tokenizer.tokenize(sentence)) != 0)]
    return frases_tokenize #retorna lista com as frases

def printBest(doc, fivebest, OriginalDocs):
    sentences =DocToSentences(OriginalDocs[doc])
    for i in fivebest[doc]:
        print(sentences[int(i)])

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
        aux.append(sentences[i])
        return aux


def stringToTerms(text, tagger1, chunker, stopwords):
    tokenizer = RegexpTokenizer(r'\w+')
    unigrams= tokenizer.tokenize(text) # todas as palavras do texto
    unigrams = [word.lower() for word in unigrams if word.lower() not in stopwords]
    #get noun phrases from text
    # #if because of sentences conposed only by stop words 'nao so isso'
    noun_phrases=[]
    if(len(unigrams))!=0:
        noun_phrases=extract(unigrams, tagger1, chunker,stopwords)
    bigrams = ngrams(unigrams, 2)
    text_bigrams = [' '.join(grams) for grams in bigrams]
    candidates = unigrams + text_bigrams +noun_phrases
    candidates=[word.lower() for word in set(candidates) if (word.lower() not in stopwords and 2 <= len(word))]
    return candidates# todas as palavras do texto

def extract(unigrams, tagger1, chunker,stopwords):
    stopwords=stopwords
    postoks = tagger1.tag(unigrams)
    tree = chunker.parse(postoks)
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='np'):
            yield subtree.leaves()
    def acceptable_word(word):
        accepted = bool(2 <= len(word) <= 40 and word.lower() not in stopwords)
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


def getFiveBest(sentences_scores, resumelen):
    five_best=[]
    for x in range(0,resumelen):
        maxSent=max(sentences_scores.keys(), key=(lambda key: sentences_scores[key]))
        five_best.append(maxSent)
        del sentences_scores[maxSent]
    five_best.sort()
    return five_best

def sumMultiPesos(doc,sentence1, sentence2, tfIdf):
    value=0
    aux = set(tfIdf[doc][sentence1].keys()).intersection(tfIdf[doc][sentence2].keys())
    for term in aux:
        value += tfIdf[doc][sentence1][term] * tfIdf[doc][sentence2][term]
    return value
def sumMultiPesosDoc(doc,sentence,pesosDoc,tfIdf,  invertedListDoc):
    value=0
    maxTf=getFqMaxDoc(doc,  invertedListDoc)# Tf maximo dos termos no documento
    for term in tfIdf[doc][sentence]:
        value+=(tfIdf[doc][sentence][term] * pesosDoc[term])
        #print("sumMultiPesos "+ str(sentence)+" "+term+" "+str(value))
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
        #print("\n\n\n\n")
       # print("doc "+doc)
        for sentence in docSentenceTerm[doc]:
           # print("doc " + doc+" sentence "+str(sentence))
            tfIdf[doc][sentence]=dict()
            maxi=maxTermfq(doc,sentence, invertedList)
            for term in set(docSentenceTerm[doc][sentence]):
               # print("doc " + doc + " sentence " + str(sentence)+ " term "+str(term))
                _idf=idf(term,doc, OriginalDocs, invertedList)
                tf=invertedList[term][doc][sentence]
                value=(tf/maxi)*_idf
                tfIdf[doc][sentence][term]=value
    return tfIdf



def setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker,stopwords,numTermsDoc,numTermsDocSentence, PATH_TEXT):
    for doc in docs:
        numTermsDoc[doc]=0
        numTermsDocSentence[doc]=dict()
        f2 = open(PATH_TEXT + doc, "r")
        text = f2.read()
        OriginalDocs[doc] = text
        text = text.lower()
        sentences = DocToSentences(text)
        invertedListDoc[doc] = dict()
        docSentenceTerm[doc] = dict()
        sentence_counter = 0
        for sentence in sentences:
            #if len(sentence) != 0:
            aux_terms = stringToTerms(sentence, tagger1, chunker,stopwords)
            #if (len(aux_terms) != 0):
            numTermsDoc[doc]+=len(aux_terms)
            numTermsDocSentence[doc][sentence_counter]=len(aux_terms)
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

def saveResumes(resumes, PATH_AUTO_IDEAL_EXTRACTIVES):
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


def getOriginalSentence(doc,idexs,OriginalDocs):
	#global OriginalDocs
	#f2 = open(PATH_SOURCE_TEXT+doc, "r")
	#f2 = open(doc, "r")
	text = OriginalDocs[doc]
	sentences = DocToSentences(text)
	#print("doc: "+doc+ " tem "+str(len(sentences)))
	aux=[]
	for i in idexs:
	#   print("This is the value of i: " + str(i-1))
		aux.append(sentences[i])
	return aux

########deboraaa#########
def calculateScoreOfsentences(doc, tfIdf, invertedListDoc, OriginalDocs, invertedList):
    pesosDoc = calcpesoTermoDoc(doc, tfIdf, invertedListDoc, OriginalDocs, invertedList)
    sentences_scores = dict()
    sqrt_some_squares_doc = sqrtSomeSquaresDoc(pesosDoc)
    for sentence in tfIdf[doc]:
        sqrt_some_squares = sqrtSomeSquares(doc, sentence, tfIdf)
        soma_mult_pesos = sumMultiPesosDoc(doc,sentence,pesosDoc,tfIdf,  invertedListDoc)
        aux=sqrt_some_squares * sqrt_some_squares_doc
        if(aux!=0):
            sentences_scores[sentence] = (soma_mult_pesos) / (aux)
        else:
            sentences_scores[sentence]=0
    return sentences_scores


def calcpesoTermoDoc(doc,tfIdf, invertedListDoc, OriginalDocs, invertedList):
    pesosDoc= dict()
    maxTf=getFqMaxDoc(doc, invertedListDoc)
    for sentence in tfIdf[doc]:
        for term in tfIdf[doc][sentence]:
            pesosDoc[term]=((getFqTermDoc(term,doc,invertedListDoc)/maxTf) * idf(term,doc, OriginalDocs, invertedList))
    return pesosDoc

def getFqTermDoc(term,doc,invertedListDoc):
    value=0
    if term in invertedListDoc[doc]:
        for sentence in invertedListDoc[doc][term]:
            value+= invertedListDoc[doc][term][sentence]
    return value


def getSentencesScoreDoc(docs,docSentenceTerm, invertedList, OriginalDocs, invertedListDoc):
    tfIdf1 = setTfIdf(docSentenceTerm, invertedList, OriginalDocs)
    scoresDocs = dict()
    for doc in docs:
        scoresDocs[doc] = calculateScoreOfsentences(doc, tfIdf1, invertedListDoc, OriginalDocs, invertedList)
    return scoresDocs


def getNounFrases(text,tagger1, chunker, stopwords):
    tokenizer = RegexpTokenizer(r'\w+')
    unigrams = tokenizer.tokenize(text)  # todas as palavras do texto
    unigrams = [word.lower() for word in unigrams if word.lower() not in stopwords]
    # get noun phrases from text
    # #if because of sentences conposed only by stop words 'nao so isso'
    noun_phrases = []
    if (len(unigrams)) != 0:
        noun_phrases = extract(unigrams, tagger1, chunker, stopwords)
    return noun_phrases

#####################functions pagerank e ralacionadas do proj 2 ex2 e 3
def createGraphCosSimilarity(docs, tfIdf1):
    setofGraphs = dict()
    count = 0
    # print("tfidf", tfIdf1)
    for doc in docs:
        grafo = [[0 for x in range(len(tfIdf1[doc]))] for y in range(len(tfIdf1[doc]))]
        for sentence1 in tfIdf1[doc]:
            for sentence2 in tfIdf1[doc]:
                numerador = sumMultiPesos(doc, sentence1, sentence2, tfIdf1)
                # print("numerador", numerador)
                denominador1 = sqrtSomeSquares(doc, sentence1, tfIdf1)
                # print("denominador1", denominador1)
                denominador2 = sqrtSomeSquares(doc, sentence2, tfIdf1)
                # print("denominador2", denominador2)
                aux = denominador1 * denominador2
                if (aux != 0):
                    similarity = numerador / (aux)
                else:
                    similarity = 0
                    # print("similarity", similarity)
                if similarity > TRESHOLD:
                    grafo[sentence1][sentence2] = similarity
                    # for sentence1 in tfIdf1[doc]:
                    # aux = " "
                    # for sentence2 in tfIdf1[doc]:
                    #  aux+= " " + str(grafo[sentence1][sentence2])
                    # print(aux)
        setofGraphs[doc] = grafo
    return setofGraphs


def getPr0(numsentences):
    Po = 1 / numsentences
    probpre = dict()
    for i in range(numsentences):
        probpre[i] = Po
    return probpre


def getPr0BasedSentencePosition(numsentences):
    Po = 1 / numsentences
    probpre = dict()
    som = 0
    for i in range(numsentences):
        som += Po * (numsentences / (i + 1))
        probpre[i] = Po * (numsentences / (i + 1))

    for i in probpre:
        probpre[i] = probpre[i] / som
    return probpre


def getPr0BasedSentenceWeigth(numsentences, sentenceScore):
    Po = 1 / numsentences
    probpre = dict()
    som = 0
    for i in range(numsentences):
        som += Po * sentenceScore[i]
        probpre[i] = Po * sentenceScore[i]

    for i in probpre:
        probpre[i] = probpre[i] / som
    return probpre


def pageRank(numsentences, graph, Pr0):
    d = 0.15
    # probpre=getPr0(numsentences)
    # probpre=getPr0BasedSentencePosition(numsentences)
    probpre = Pr0
    prior = probpre
    probpos = dict()
    for x in range(50):
        for i in range(numsentences):
            aux = somatorioPriors(prior, i, graph, numsentences)
            v = 0
            if (aux != 0):
                v = prior[i] / aux
            probpos[i] = (d * (v)) + (1 - d) * (somatorioPesos(probpre, i, graph, numsentences))
        probpre = probpos
    return probpos


def somatorioPriors(prior, i, graph, numsentences):
    value = 0
    for j in range(numsentences):
        if graph[i][j] > 0:
            value += graph[i][j]
    return value


def somatorioPesos(probpre, i, graph, numsentences):
    value = 0
    for j in range(numsentences):
        counter = 0
        if graph[i][j] > 0:
            for k in range(numsentences):
                if graph[j][k] > 0:
                    counter += graph[j][k]
            value = value + (probpre[j] * graph[i][j] / counter)
    return value


def creatGrafsNounFrases(OriginalDocs, tagger1, chunker, stopwords):
    # text is the text of the original doc
    setofGraphs = dict()
    count = 0
    # print("tfidf", tfIdf1)
    for doc in OriginalDocs:
        text = OriginalDocs[doc]
        text = text.lower()
        sentences = DocToSentences(text)
        grafo = [[0 for i in range(len(sentences))] for j in range(len(sentences))]
        x = 0
        y = 0
        for sentence1 in sentences:
            nounPhrases1 = getNounFrases(sentence1, tagger1, chunker, stopwords)
            y = 0
            for sentence2 in sentences:
                nounPhrases2 = getNounFrases(sentence2, tagger1, chunker, stopwords)
                grafo[x][y] = len(set(nounPhrases1).intersection(nounPhrases2))
                y += 1
            x += 1
        setofGraphs[doc] = grafo
    return setofGraphs


def getPr0BasedNumTermsFrase(numsentences, numTermsDoc, numTermsSentence):
    probpre = dict()
    for i in range(numsentences):
        probpre[i] = numTermsSentence[i] / numTermsDoc
    return probpre
