from nltk.tokenize import sent_tokenize, word_tokenize

filedata = open("teste.tx").read()

sentences = sent_tokenize(filedata)
words = []

for lines in sentences:
	words.append(word_tokenize(lines))
