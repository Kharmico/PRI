from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.chunk import RegexpParser
from nltk import ngrams
import os, nltk, string, itertools


 #1. we obtain the unigrams
    unigrams = nltk.word_tokenize(text)
    #2. we get the bigrams
    bigrams = ngrams(unigrams, 2)


 # POS Tagging each sentence
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    grammar = "NP: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))

    # This is the right way to go, bigrams and trigrams will not consider inter-phrase tokens   
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                      for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]
