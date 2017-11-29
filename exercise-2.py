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
                similarity = numerador / (denominador1*denominador2)
               # print("similarity", similarity)
                if similarity > TRESHOLD:
                    grafo[sentence1-1][sentence2-1]=similarity
        for sentence1 in tfIdf1[doc]:
            aux = " "
            for sentence2 in tfIdf1[doc]:
                aux+= " " + str(grafo[sentence1 - 1][sentence2 - 1])
            #print(aux)
        setofGraphs[doc]=grafo
    return setofGraphs



def pageRank(numsentences, graph ):
    d = 0.15
    Po = 1/ numsentences
    #probpre = [Po for x in range(numsentences)]
   # probpos = [0 for x in range(numsentences)]
    probpre=dict()
    probpos =dict()
    for i in range(numsentences):
        probpre[i+1]=Po
        probpos[i+1]=0
   # for x in range(50):
    for i in range(numsentences):
        probpos[i+1] =  (d / numsentences) + (1-d) * (somatorio(probpre, i, graph, numsentences))
    probpre = probpos
    return probpos

def somatorio(probpre, i, graph, numsentences):
    value = 0
    for j in range(numsentences) :
        counter= 0
        if graph[i][j] > 0:
            for k in range(numsentences):
                if graph[j][k] > 0:
                    counter +=1
            value =value+ (probpre[j+1] / counter)
    return value



def main():
    global docs
    resume1 = dict()
    num_docs = len(docs)
    mean_avg_precision = 0
    tagger1 = getTagger()
    chunker = getChunker()
    extracted = saveResumes(resumes)
    stopwords= getStopWords()
   # print("Set inverted List")
    setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker,stopwords)
    tfIdf1 = setTfIdf(docSentenceTerm, invertedList, OriginalDocs)
    #print("Set of Graphas")
    setofGraphs =  createGraph(docs,tfIdf1)
    #print("begin loop")
    for doc,graph in setofGraphs.items():
        #print("doc "+str())
        numPhrases=len(tfIdf1[doc].keys())
        pagescore = pageRank(numPhrases, graph)
        resume1[doc] = getOriginalSentence(doc,getFiveBest(pagescore, 5),OriginalDocs)
        mean_avg_precision += calc_avg_doc(resume1[doc], extracted[doc])
    mean_avg_precision = mean_avg_precision/ num_docs
    print("MAP : " + str(mean_avg_precision))


main()






