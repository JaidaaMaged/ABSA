import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

class AspectExtractor(object):

    grammar = ('''
        NP: {<DT>?<JJ>*<NN>} # NP
        ''')
    chunkParser = nltk.RegexpParser(grammar)
    POS_Chunked_Reviews = []
    one_review = []

    def __init__(self):
        pass

    def extract_nouns(self,sentence):
        """
                INPUT: sentence
                OUTPUT: list of lists of strings
                Given a review sentence, return the aspects
                """
        tagged = nltk.pos_tag(sentence)
        one_review_sentence = self.chunkParser.parse(tagged)
        sent_nouns = self.extract_NPs(one_review_sentence)
        filtered_sent_nouns = self.remove_stopwords_stemming(sent_nouns)
        return filtered_sent_nouns

    def extract_NPs(self, tree):
        """"Returns a list of nouns & noun phrases of the given review sentence tree"""
        x = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            nodes = subtree.leaves()
            for q in range(len(nodes)):
                part_chunk = nodes[q]
                extracted_noun = part_chunk[0]
                x.append(extracted_noun)


        return x



    def remove_stopwords_stemming(self, Nouns):

        """"
        TAKES UNFIlTERED NOUNS & RETURNS FILTERED NOUNS
        """
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        filtered_sentence = []


        for w in Nouns:
            if w not in stop_words:
                x = ps.stem(w)
                filtered_sentence.append(x)

        return filtered_sentence
    def extract_freq_nouns (self,temp):
        te = TransactionEncoder()

        te_ary = te.fit( temp ).transform( temp )

        df = pd.DataFrame( te_ary, columns=te.columns_ )

        frequent_itemsets = apriori( df, min_support=0.01, use_colnames=True)

        print( frequent_itemsets)










