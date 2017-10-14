import re
from sklearn.feature_extraction.text import TfidfVectorizer
regexExpression = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[a-zA-Z]+"

def stringToWords(text):
	return re.findall(regexExpression, text)

def readfile(filename):
	f2=open(filename,"r")
	text=f2.read()
	tempText=stringToWords(text)
	print(tempText)

readfile("teste.txt")

