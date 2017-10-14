import re
from sklearn.feature_extraction.text import TfidfVectorizer
regexWords = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[a-zA-Z]+"
regexSentences="(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
terms=[]
sentences =dict()
def stringToTerms(text):
	return re.findall(regexWords, text)
def stringToDictOfSentences(text):
	aux=[]
	aux=re.split(regexSentences, text)
	global sentences
	i=0
	for s in aux:
		print(i)
		sentences[i]=stringToTerms(s)
		i+=1



def getTermFromText(text):
	global terms
	auxTerms=stringToTerms(text)
	for t in auxTerms:
		if t not in terms:
			terms+=[t.lower()]
	



def readfile(filename):
	global terms,sentences
	f2=open(filename,"r")
	text=f2.read()
	getTermFromText(text)
	print("termos:")
	print(terms)
	stringToDictOfSentences(text)
	print("frazes")
	print(sentences)
	#ntempText=stringToSentences(text)


readfile("teste.txt")

