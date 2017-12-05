import os
from functions import *

# from functions import printBest, sqrtSomeSquares,  getResume, sumMultiPesos, setTfIdf, setInvertedList, calc_avg_doc, saveResumes, getChunker, getTagger

PATH_TEXT = './teste/'
PATH_SOURCE_TEXT = './SourceTextWithTitle/'
PATH_MANUAL_SUMMARIES = './ManualSummaries/'
PATH_AUTO_IDEAL_EXTRACTIVES = './AutoIdealExtractives/'
RESUME_LEN = 5


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

def main():
    global docs
    resume1 = dict()
    resume2 = dict()
    resume3 = dict()
    resume4 = dict()
    resume5= dict()
    num_docs = len(docs)
    mean_avg_precision1 = 0
    mean_avg_precision2 = 0
    mean_avg_precision3 = 0
    mean_avg_precision4 = 0
    mean_avg_precision5 = 0
    tagger1 = getTagger()
    chunker = getChunker()
    extracted = saveResumes(resumes, PATH_AUTO_IDEAL_EXTRACTIVES)
    stopwords = getStopWords()
    numTermsDoc = dict()
    numTermsDocSentence = dict()
    setInvertedList(docs, OriginalDocs, invertedListDoc, docSentenceTerm, invertedList, tagger1, chunker, stopwords, numTermsDoc, numTermsDocSentence, PATH_TEXT)
    tfIdf1 = setTfIdf(docSentenceTerm, invertedList, OriginalDocs)
    setofGraphs1 =  createGraphCosSimilarity(docs, tfIdf1)
    setofGraphs2 = creatGrafsNounFrases(OriginalDocs, tagger1, chunker, stopwords)
    sentencesScores = getSentencesScoreDoc(docs, docSentenceTerm, invertedList, OriginalDocs, invertedListDoc, RESUME_LEN)
    for doc, graph in setofGraphs1.items():
        # print("doc "+str())
        numPhrases = len(tfIdf1[doc].keys())
        pr0=getPr0BasedSentencePosition(numPhrases)
        pr1=getPr0(numPhrases)
        pr2 = getPr0BasedNumTermsFrase(numPhrases, numTermsDoc[doc], numTermsDocSentence[doc])
        pr3=getPr0BasedSentenceWeigth(numPhrases,sentencesScores[doc])
        pagescore11 = pageRank(numPhrases, setofGraphs1[doc], pr0)
       # pagescore12 = pageRank(numPhrases, setofGraphs2[doc], pr0)
        pagescore21 = pageRank(numPhrases, setofGraphs1[doc], pr1)
        pagescore22 = pageRank(numPhrases, setofGraphs2[doc], pr1)
        pagescore31 = pageRank(numPhrases, setofGraphs1[doc], pr2)
        resume1[doc] = getOriginalSentence(doc, getFiveBest(pagescore11, RESUME_LEN), OriginalDocs)
       # resume2[doc] = getOriginalSentence(doc, getFiveBest(pagescore12, RESUME_LEN), OriginalDocs)
        resume3[doc] = getOriginalSentence(doc, getFiveBest(pagescore21, RESUME_LEN), OriginalDocs)
        #resume4[doc] = getOriginalSentence(doc, getFiveBest(pagescore22, RESUME_LEN), OriginalDocs)
        resume5[doc] = getOriginalSentence(doc, getFiveBest(pagescore31, RESUME_LEN), OriginalDocs)
        mean_avg_precision1 += calc_avg_doc(resume1[doc], extracted[doc])
       # mean_avg_precision2 += calc_avg_doc(resume2[doc], extracted[doc])
        mean_avg_precision3 += calc_avg_doc(resume3[doc], extracted[doc])
      #  mean_avg_precision4 += calc_avg_doc(resume4[doc], extracted[doc])
        mean_avg_precision5 += calc_avg_doc(resume5[doc], extracted[doc])


    mean_avg_precision1 = mean_avg_precision1 / num_docs
    #mean_avg_precision2 = mean_avg_precision2 / num_docs
    mean_avg_precision3 = mean_avg_precision3 / num_docs
   # mean_avg_precision4 = mean_avg_precision4 / num_docs

    print("Non-uniform prior weights based on the cosine similarity towards the entire document,leverating either TF-IDF")
    print("MAP1: " + str(mean_avg_precision1))
   # print("MAP2 : " + str(mean_avg_precision2))
    print("Non-uniform prior weights with basis on the position of the sentence in the document")
    print("MAP3 : " + str(mean_avg_precision3))
   # print("Non-uniform prior weights with basis on the position of the sentence in the document + Noun Phrases Graph")
   # print("MAP4 : " + str(mean_avg_precision4))
    print("Edge weights based on the number of sentences terms")
    print("MAP5 : " + str(mean_avg_precision5))

main()






