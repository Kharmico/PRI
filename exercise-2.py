import os
from functions import *
#from functions import printBest, sqrtSomeSquares,  getResume, sumMultiPesos, setTfIdf, setInvertedList, calc_avg_doc, saveResumes, getChunker, getTagger

PATH_TEXT = './teste/'
PATH_SOURCE_TEXT = './SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES = './ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES = './AutoIdealExtractives/'
RESUME_LEN = 5
TRESHOLD = 0.15


sentInExtResumes = 0
terms = dict()
invertedList = dict()
invertedListDoc = dict()
docSentenceTerm = dict()
docs = [f for f in os.listdir(PATH_TEXT)]
OriginalDocs = dict()
resumes = [f for f in os.listdir(PATH_AUTO_IDEAL_EXTRACTIVES)]
scores = dict()
scores2 = dict()
tfIdf = dict()
tf = dict()
num_frases_termo = dict()


def createGraph(docs,tfIdf1):
    setofGraphs= dict()
    count = 0
    #print("tfidf", tfIdf1)
    for doc in docs:
        grafo = [[0 for x in range(len(tfIdf1[doc]))] for y in range(len(tfIdf1[doc]))]
        for sentence1 in tfIdf1[doc]:
            for sentence2 in tfIdf1[doc]:
                numerador = sumMultiPesos(doc,sentence1,sentence2,tfIdf1)
               # print("numerador", numerador)
                denominador1 = sqrtSomeSquares(doc,sentence1,tfIdf1)
               # print("denominador1", denominador1)
                denominador2 = sqrtSomeSquares(doc,sentence2,tfIdf1)
               # print("denominador2", denominador2)
                aux=denominador1*denominador2
                if(aux!=0):
                    similarity = numerador / (aux)
                else:
                    similarity=0
               # print("similarity", similarity)
                if similarity > TRESHOLD:
                    grafo[sentence1][sentence2]=similarity
        for sentence1 in tfIdf1[doc]:
            aux = " "
            for sentence2 in tfIdf1[doc]:
                aux+= " " + str(grafo[sentence1][sentence2])
            #print(aux)
        setofGraphs[doc]=grafo
    return setofGraphs


def getPr0(numsentences):
    Po = 1/ numsentences
    probpre=dict()
    for i in range(numsentences):
    	probpre[i]=Po
    return probpre

def getPr0BasedSentencePosition(numsentences):
    Po = 1/ numsentences
    probpre=dict()
    som=0
    for i in range(numsentences):
        som+=Po*(numsentences/(i+1))
        probpre[i]=Po*(numsentences/(i+1))
    
    for i in probpre:
        probpre[i]=probpre[i]/som
    #test
    #som=0
    #for i in probpre:
    #    som+=probpre[i]
    #if(som!=1):
    #    print("erro som= "+str(som))
    return probpre


def getPr0BasedSentenceWeigth(numsentences,sentenceScore):
    Po = 1 / numsentences
    probpre = dict()
    som = 0
    for i in range(numsentences):
        som += Po * sentenceScore[i]
        probpre[i] = Po * sentenceScore[i]

    for i in probpre:
        probpre[i] = probpre[i] / som
    # test
    # som=0
    # for i in probpre:
    #    som+=probpre[i]
    # if(som!=1):
    #    print("erro som= "+str(som))
    return probpre

def pageRank(numsentences, graph ,Pr0,pesos):
    d = 0.15
    #probpre=getPr0(numsentences)
    #probpre=getPr0BasedSentencePosition(numsentences)
    probpre=Pr0
    prior=probpre
    probpos =dict()
    for x in range(50):
    	for i in range(numsentences):
            aux=somatorioPriors(prior,i, graph, numsentences)
            v=0
            if(aux!=0):
                v=prior[i]/aux
            probpos[i] =  (d *(v)) + (1-d) * (somatorioPesos(probpre, i, graph, numsentences,pesos))
    	probpre = probpos
    return probpos
def somatorioPriors(prior,i, graph, numsentences):
    value = 0
    for j in range(numsentences) :
        if graph[i][j] > 0:
            value+=graph[i][j]
    return value


def somatorioPesos(probpre, i, graph, numsentences,pesos):
    value = 0
    for j in range(numsentences) :
        counter= 0
        if graph[i][j] > 0:
            for k in range(numsentences):
                if graph[j][k] > 0:
                    counter +=graph[j][k]
            value =value+ (probpre[j]* graph[i][j]/ counter)
    return value

def getPesosNounFrases (text,numPhrases,tagger1, chunker, stopwords):
    #text is the text of the original doc
    text = text.lower()
    sentences = DocToSentences(text)
    pesos= [[0 for i in range(numPhrases)] for j in range(numPhrases)]
    if(numPhrases!=len(sentences)):
        print("erro")
    x=0
    y=0
    for sentence1 in sentences:
        nounPhrases1 = getNounFrases(sentence1,tagger1, chunker, stopwords)
        y=0
        for sentence2 in sentences:
            nounPhrases2 = getNounFrases(sentence2, tagger1, chunker, stopwords)
            pesos[x][y]=len(set(nounPhrases1).intersection(nounPhrases2))
            y+=1
        x+=1
    return pesos


def main():
    global docs
    resume1 = dict()
    num_docs = len(docs)
    mean_avg_precision = 0
    tagger1 = getTagger()
    chunker = getChunker()
    extracted = saveResumes(resumes)
    stopwords= getStopWords()
    setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker,stopwords)
    tfIdf1 = setTfIdf(docSentenceTerm, invertedList, OriginalDocs)
    setofGraphs =  createGraph(docs,tfIdf1)
    sentencesScores=getSentencesScoreDoc(docs, docSentenceTerm, invertedList, OriginalDocs, invertedListDoc, RESUME_LEN)
    for doc,graph in setofGraphs.items():
        #print("doc "+str())
        numPhrases=len(tfIdf1[doc].keys())
        #pr0=getPr0BasedSentencePosition(numsentences)
        #pr0=getPr0(numPhrases)
        pr0=getPr0BasedSentenceWeigth(numPhrases,sentencesScores[doc])
        #pesos the similaridade entre pares de frazes já existe no grafo
        #pesos=graph
        pesos=getPesosNounFrases(OriginalDocs[doc],numPhrases,tagger1, chunker, stopwords)
        pagescore = pageRank(numPhrases, graph,pr0,pesos)
        resume1[doc] = getOriginalSentence(doc,getFiveBest(pagescore, RESUME_LEN),OriginalDocs)
        mean_avg_precision += calc_avg_doc(resume1[doc], extracted[doc])
    mean_avg_precision = mean_avg_precision/ num_docs
    print("MAP : " + str(mean_avg_precision))


main()






