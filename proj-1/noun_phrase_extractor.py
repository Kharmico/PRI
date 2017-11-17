import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
grammar = "NP: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"
chunker = nltk.RegexpParser(grammar)
stopwords = set(nltk.corpus.stopwords.words('portuguese'))

def extract(unigrams):
   
    postoks = nltk.tag.pos_tag(unigrams)
    tree = chunker.parse(postoks)
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def normalise(word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = lemmatizer.lemmatize(word)
        return word

    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(2 <= len(word) <= 40
            and word.lower() not in stopwords)
        return accepted


    def get_terms(tree):
        for leaf in leaves(tree):
            term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
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
def main():
    text = "Empresa fadada ao insucesso tem duas caras: uma real, outra para o cliente."
    tokenizer = RegexpTokenizer(r'\w+')
    unigrams= tokenizer.tokenize(text) # todas as palavras do texto 
    res=extract(unigrams)
    print(res)
main()