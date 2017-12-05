import os
from functions import *


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
    print("The operation will take a few minutes...")
    resume1 = dict()
    resume2 = dict()
    resume3 = dict()
    resume4 = dict()
    resume5= dict()
    resume6= dict()
    resume7= dict()
    resume8= dict()
    num_docs = len(docs)
    mean_avg_precision1 = 0
    mean_avg_precision2 = 0
    mean_avg_precision3 = 0
    mean_avg_precision4 = 0
    mean_avg_precision5 = 0
    mean_avg_precision6 = 0
    mean_avg_precision7 = 0
    mean_avg_precision8 = 0
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
    sentencesScores = getSentencesScoreDoc(docs, docSentenceTerm, invertedList, OriginalDocs, invertedListDoc)
    for doc, graph in setofGraphs1.items():
        # print("doc "+str())
        numPhrases = len(tfIdf1[doc].keys())
        pr0=getPr0(numPhrases) #original
        pr1=getPr0BasedSentencePosition(numPhrases)
        pr2 = getPr0BasedNumTermsFrase(numPhrases, numTermsDoc[doc], numTermsDocSentence[doc])
        pr3=getPr0BasedSentenceWeigth(numPhrases,sentencesScores[doc])
        pagescore1 = pageRank(numPhrases, setofGraphs1[doc], pr0)
        pagescore6 = pageRank(numPhrases, setofGraphs2[doc], pr0)
        pagescore2 = pageRank(numPhrases, setofGraphs1[doc], pr1)
        pagescore7 = pageRank(numPhrases, setofGraphs2[doc], pr1)
        pagescore3 = pageRank(numPhrases, setofGraphs1[doc], pr2)
        pagescore8 = pageRank(numPhrases, setofGraphs2[doc], pr2)
        pagescore4= pageRank(numPhrases, setofGraphs1[doc], pr3)
        pagescore5= pageRank(numPhrases, setofGraphs2[doc], pr3)
        resume1[doc] = getOriginalSentence(doc, getFiveBest(pagescore1, RESUME_LEN), OriginalDocs)
        resume2[doc] = getOriginalSentence(doc, getFiveBest(pagescore2, RESUME_LEN), OriginalDocs)
        resume3[doc] = getOriginalSentence(doc, getFiveBest(pagescore3, RESUME_LEN), OriginalDocs)
        resume4[doc] = getOriginalSentence(doc, getFiveBest(pagescore4, RESUME_LEN), OriginalDocs)
        resume5[doc] = getOriginalSentence(doc, getFiveBest(pagescore5, RESUME_LEN), OriginalDocs)
        resume6[doc] = getOriginalSentence(doc, getFiveBest(pagescore6, RESUME_LEN), OriginalDocs)
        resume7[doc] = getOriginalSentence(doc, getFiveBest(pagescore7, RESUME_LEN), OriginalDocs)
        resume8[doc] = getOriginalSentence(doc, getFiveBest(pagescore8, RESUME_LEN), OriginalDocs)

        mean_avg_precision1 += calc_avg_doc(resume1[doc], extracted[doc])
        mean_avg_precision2 += calc_avg_doc(resume2[doc], extracted[doc])
        mean_avg_precision3 += calc_avg_doc(resume3[doc], extracted[doc])
        mean_avg_precision4 += calc_avg_doc(resume4[doc], extracted[doc])
        mean_avg_precision5 += calc_avg_doc(resume5[doc], extracted[doc])
        mean_avg_precision6 += calc_avg_doc(resume6[doc], extracted[doc])
        mean_avg_precision7 += calc_avg_doc(resume7[doc], extracted[doc])
        mean_avg_precision8 += calc_avg_doc(resume8[doc], extracted[doc])




    mean_avg_precision1 = mean_avg_precision1 / num_docs
    mean_avg_precision2 = mean_avg_precision2 / num_docs
    mean_avg_precision3 = mean_avg_precision3 / num_docs
    mean_avg_precision4 = mean_avg_precision4 / num_docs
    mean_avg_precision5 = mean_avg_precision5 / num_docs
    mean_avg_precision6 = mean_avg_precision6 / num_docs
    mean_avg_precision7 = mean_avg_precision7 / num_docs
    mean_avg_precision8 = mean_avg_precision8 / num_docs



    print("MAP1: " + str(mean_avg_precision1))
    print("MAP2 : " + str(mean_avg_precision2))
    print("MAP3 : " + str(mean_avg_precision3))
    print("MAP4 : " + str(mean_avg_precision4))
    print("MAP5 : " + str(mean_avg_precision5))
    print("MAP6 : " + str(mean_avg_precision6))
    print("MAP7 : " + str(mean_avg_precision7))
    print("MAP8 : " + str(mean_avg_precision8))


main()











