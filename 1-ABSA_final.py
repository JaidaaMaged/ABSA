import pandas as pd
import re
# from write_in_pdf import  PDF
from fpdf import FPDF
import image

desired_width = 320
pd.set_option( 'display.width', desired_width )
#read dataset & convert it to a list
data = pd.read_csv( "Amazon_Unlocked_Mobile.csv" )
reviews = []
reviews = data['Reviews'].tolist()

review_sentences = []

for i in reviews:

    #split reviews to sentences
    review_sentences.append( re.split( r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', str( i ) ) )


import csv
import nltk
from autocorrect import spell
from nltk.corpus import product_reviews_1
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#smaller dataset from Hu & Liu Paper
"""camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
review = camera_reviews[0]
review.sents()[0]
"""

grammar = ('''
    NP: {<DT>?<JJ>*<NN>} # NP
    ''')
chunkParser = nltk.RegexpParser( grammar )
POS_Chunked_Reviews = []
one_review = []
#8400 is the number of reviews limited to the parser
for i in range( 8400 ):

    one_review = []
    for j in range( len( review_sentences[i] ) ):
        #Convert sent to list to input to POS tagger
        sent_to_be_tagged = review_sentences[i][j].split( " " )
        #Remove empty lists.
        while '' in sent_to_be_tagged:
            sent_to_be_tagged.remove( '' )
        tagged = nltk.pos_tag( sent_to_be_tagged )
        one_review_sentence = chunkParser.parse( tagged )
        x = []
        for subtree in one_review_sentence.subtrees( filter=lambda t: t.label() == 'NP' ):
            #wa is a list containing the word of the noun phrase + its tag
            wa = subtree.leaves()
            for q in range( len( wa ) ):
                part_chunk = wa[q]
                if (part_chunk[1] == 'JJ'):

                    continue
                final = part_chunk[0]
                #x is the list of the noun phrases in each review sentence
                x.append( final )
        #one_review is the list of noun phrases in the entire review
        one_review.append( x )
    POS_Chunked_Reviews.append( one_review )
stop_words = nltk.corpus.stopwords.words( 'english' )
newStopWords = ['everything', 'something', 'anyone', 'thing', 'lot']
stop_words.extend( newStopWords )
Filtered_Reviews = []
ps = PorterStemmer()
Reviews = []
Filtered_Review = []
Temp = []
#Remove stop words from list of noun phrases
with open( 'transaction_file.csv', mode='w', newline='' ) as transaction_file:
    transaction_writer = csv.writer( transaction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
    for i in range( len( POS_Chunked_Reviews ) ):
        Filtered_Review = []

        for j in range( len( POS_Chunked_Reviews[i] ) ):
            word_tokens = POS_Chunked_Reviews[i][j]
            filtered_sentence = []

            for w in word_tokens:

                if w.lower() not in stop_words:
                    """" filtered_sentence.append(ps.stem(w))"""
                    print( "not a stop word: ", w )
                    filtered_sentence.append( w )
            Filtered_Review.append( filtered_sentence )
            Temp.append( filtered_sentence )
        Reviews.append( Filtered_Review )

from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()

te_ary = te.fit( Temp ).transform( Temp, sparse= True )
df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

#df = pd.DataFrame( te_ary, columns=te.columns_ )
frequent_itemsets = apriori( df, min_support=0.01, use_colnames=True )

itemsets = frequent_itemsets.iloc[:, 1]
frequent_itemsets_list = [list( x ) for x in itemsets]

support = frequent_itemsets.iloc[:, 0]




# redundancy pruning:
single = []
for i in range( len( frequent_itemsets_list ) ):

    if len( frequent_itemsets_list[i] ) >= 2:
        break
    search = frequent_itemsets_list[i]
    sum = 0

    for j in range( i + 1, len( frequent_itemsets_list ) ):

        for z in range( len( frequent_itemsets_list[j] ) ):

            if frequent_itemsets_list[j][z] == search[0]:
                #how many times it came as a part of a multi-aspect
                sum = sum + support[j]
    #length of temp, is the number of review sentences
    support_single = support[i] * len( Temp )
    support_multi = sum * len( Temp )
    diff = support_single - support_multi
    if (diff > 400):
        #then I'll consider this aspect as a single aspect on its own, not only as a part of a multi-aspect
        single.append( search )



# compactness pruning:
from textblob import Word

cleaned = list()
multi_asps = list()
cleaned = cleaned + single
for i in range( len( frequent_itemsets_list ) ):
    "If single aspect - continue"
    if len( frequent_itemsets_list[i] ) == 1:
        continue
    count = 0
    no_of_words = 0
    Phrase = []
    for j in range( len( frequent_itemsets_list[i] ) ):
        multi_asps.append( frequent_itemsets_list[i] )
        "Number of words in multi-aspect term"
        no_of_words += 1
        Phrase.append( frequent_itemsets_list[i][j] )
        # Count the number of small words and words without an English definition
        if len( frequent_itemsets_list[i][j] ) <= 2 or (not Word( frequent_itemsets_list[i][j] ).definitions):
            count += 1

    if count < no_of_words * 0.6:
        """"cleaned.append(Phrase)"""
        concate = []
        concate.append( Phrase[0] + ' ' + Phrase[1] )
        cleaned.append( concate )
        concate = []
        concate.append( Phrase[1] + ' ' + Phrase[0] )
        cleaned.append( concate )
#Final list of aspects after pruning from the frequency based approach
len( cleaned )



import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from textblob import TextBlob

dep_parser = CoreNLPDependencyParser( url='http://localhost:9000' )
import string


def PairExtract_adjective(i, triples, dep, feature_opinion_pairs, adjective, index):
    if (dep[1] == 'NN' or dep[1] == 'NNS' or dep[1] == 'NNP' or dep[1] == 'NNPS'):
        # rule 1
        pair = []
        pair.append( i )
        pair.append( dep[0] )
        pair.append( adjective )
        feature_opinion_pairs.append( pair )
    elif (dep[1] != 'NN' and dep[1] != 'NNS' and dep[1] != 'NNP' and dep[1] != 'NNPS'):
        for governor, dep, dependent in triples[index:]:
            if (dep == 'dobj'):
                # Rule 2
                pair = []
                pair.append( i )
                pair.append( dependent[0] )
                pair.append( adjective )
                feature_opinion_pairs.append( pair )
                break


def PairExtract_verb(i, triples, noun_exists, rmod, noun, feature_opinion_pairs, verb, index):
    for governor, dep, dependent in triples[index:]:
        if (dep == 'dobj'):
            # Rule 3
            pair = []
            pair.append( i )
            pair.append( dependent[0] )
            pair.append( verb )
            feature_opinion_pairs.append( pair )
            break
        elif (dep == 'xcomp' or dep == 'advmod'):
            pair = []
            pair.append( i )
            if (noun_exists == True):
                pair.append( rmod )

            else:
                # Rule 4 using xcomp
                pair.append( noun[0] )

            pair.append( dependent[0] )

            feature_opinion_pairs.append( pair )
            break


def PairExtract_amod(i, triples, dependent, feature_opinion_pairs, governor, index):
    aspect = governor[0]
    tag = governor[1]
    adj = dependent[0]
    pair = []
    pair.append( i )
    pair.append( governor[0] )
    pair.append( dependent[0] )
    feature_opinion_pairs.append( pair )
    for governor, dep, dependent in triples[index:]:
        if (dep == 'conj' and (tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS')):
            if ((dependent[1] == 'NN' or dependent[1] == 'NNS' or dependent[1] == 'NNP' or dependent[1] == 'NNPS') and (
                    governor[1] == 'NN' or governor[1] == 'NNS' or governor[1] == 'NNP' or governor[1] == 'NNPS')):
                pair = []
                pair.append( i )
                pair.append( dependent[0] )
                pair.append( adj )
                feature_opinion_pairs.append( pair )
            else:
                pair = []
                pair.append( i )
                pair.append( aspect )
                pair.append( dependent[0] )
                feature_opinion_pairs.append( pair )


feature_opinion_pairs = []
pair = []

noun = " "
k = 0
for i in range( 8400 ):

    for j in range( len( review_sentences[i] ) ):

        str2 = ''.join( review_sentences[i][j] )

        str1 = str2.translate( str.maketrans( '', '', string.punctuation ) )

        if (str1 == '' or str1 == ' '):
            continue
        parses = dep_parser.parse( str1.split() )

        noun_exists = False
        for parse in parses:


            triples = list( parse.triples() )

            for governor, dep, dependent in triples:
                lista = governor, dep, dependent
                """
                print(governor, dep, dependent)
                """
                index = triples.index( lista )
                dependencies = list( dep )

                if (dep == 'acl:relcl'):
                    if (triples[index - 1][1] == "compound"):
                        print("inside rcl", triples[index - 1][2][0] + " " + triples[index - 1][0][0])
                        # concatenate governor and dependent of compound relation
                        noun = triples[index - 1][2][0] + " " + triples[index - 1][0][0]
                        noun_exists = True
                    else:
                        noun = governor[0]
                        noun_exists = True

                if (dep == 'nsubj'):

                    if (governor[1] == 'JJ' or governor[1] == 'JJR' or governor[1] == 'JJS'):

                        PairExtract_adjective( i, triples, dependent, feature_opinion_pairs, governor[0], index )
                    elif (governor[1] == 'VBP' or governor[1] == 'VBZ' or governor[1] == 'VBD' or governor[
                        1] == 'VBN' or governor[1] == 'VB' or governor[1] == 'VBG'):


                        PairExtract_verb( i, triples, noun_exists, noun, dependent, feature_opinion_pairs, governor[0],
                                          index )
                    elif ((governor[1] == 'NN' or governor[1] == 'NNS' or governor[1] == 'NNP' or governor[
                        1] == 'NNPS') and (
                                  dependent[1] == 'NN' or dependent[1] == 'NNS' or dependent[1] == 'NNP' or dependent[
                              1] == 'NNPS')):

                        pair = []
                        pair.append( i )
                        pair.append( dependent[0] )
                        pair.append( governor[0] )
                        feature_opinion_pairs.append( pair )
                if (dep == 'amod'):
                    if (governor[1] == 'NN' or governor[1] == 'NNS' or governor[1] == 'NNP' or governor[1] == 'NNPS'):
                        PairExtract_amod( i, triples, dependent, feature_opinion_pairs, governor, index )
                if (dep == 'nmod'):

                    pair = []
                    pair.append( i )
                    pair.append( dependent[0] )
                    pair.append( governor[0] )
                    feature_opinion_pairs.append( pair )

                if (dep == 'compound'):
                    pair = feature_opinion_pairs[-1]
                    if (pair[1] == governor[0]):
                        pair[1] = dependent[0] + ' ' + governor[0]
                        feature_opinion_pairs[-1] = pair
                if (dep == 'neg'):
                    pair = feature_opinion_pairs[-1]
                    if(pair[2] == governor[0]):
                        pair[2] = "not " + pair[2]
                        feature_opinion_pairs[-1] = pair


opinion_pairs = []
for i in feature_opinion_pairs:
    word = i[1]
    """word = ps.stem(word)"""

    for sublist in cleaned:

        """

        print( "word " , word )
        """
        if sublist[0] == word:
            opinion_pairs.append( i )
            break

for z in opinion_pairs:
    print( "Pairs", z )

print( len( opinion_pairs ) )

polarity_opinion_pairs = []
for x in opinion_pairs:
    pol = TextBlob( x[2] )
    if pol.sentiment.polarity > 0:
        x.append( "positive" )
        polarity_opinion_pairs.append( x )
    elif pol.sentiment.polarity < 0:
        x.append( "negative" )
        polarity_opinion_pairs.append( x )
    else:
        continue

for t in polarity_opinion_pairs:
    print( t[0] )
    print( t[1] )
    print( t[2] )
    print( t[3] )
print( len( polarity_opinion_pairs ) )
print( "CLEANED" )
print( cleaned )

with open( 'aspect_file.csv', mode='w', newline='' ) as aspect_file:
    aspect_writer = csv.writer( aspect_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
    aspects = []

    for h in polarity_opinion_pairs:
        aspects.append( h[1] )
    print( aspects )
    aspects = list( dict.fromkeys( aspects ) )
    aspects.append( "label" )
    print( aspects )
    aspect_writer.writerow( aspects )
    counter_pos = [0] * (len( aspects ) - 1)
    counter_neg = [0] * (len( aspects ) - 1)
    for x in range( 8400 ):
        flag = False
        row = [None] * (len( aspects ) - 1)

        pos = 0
        neg = 0
        for b in polarity_opinion_pairs:

            if (b[0] == x):
                if (b[3] == 'positive'):
                    pos += 1
                    row[aspects.index( b[1] )] = 1
                    counter_pos[aspects.index( b[1] )] += 1
                if (b[3] == 'negative'):
                    neg += 1
                    row[aspects.index( b[1] )] = 0
                    counter_neg[aspects.index( b[1] )] += 1
        for q in range( len( row ) ):

            if (row[q] == None):
                row[q] = 2
        if (pos > neg):
            row.append( "positive" )
        elif (neg > pos):
            row.append( "negative" )

        elif (pos == 0 and neg == 0):
            flag = True
        elif (pos == neg):
            row.append( "negative" )
        if flag == False:

            aspect_writer.writerow( row )
        print( row )
    for z in range( len( aspects ) - 1 ):
        print( aspects[z], " ", counter_pos[z], " ", counter_neg[z] )

print( aspects )
print( counter_neg )
print( counter_neg )

pos_pairs = []
neg_pairs = []
for k in polarity_opinion_pairs:
    if (k[3] == 'positive'):
        pos_pairs.append( k )
        print( "append positive" )
    if (k[3] == 'negative'):
        neg_pairs.append( k )
        print( "append negative" )
print( "pos _ pairs", pos_pairs )
review_pdf = []
for h in aspects:

    c_pos = 0
    c_neg = 0
    for i in pos_pairs:

        if (h == i[1] and c_pos < 3 and reviews[i[0]] not in review_pdf):
            review_aspect = []
            review_aspect.append( h )
            review_aspect.append( "Positive" )
            review_aspect.append( reviews[i[0]] )
            c_pos += 1
            review_pdf.append( review_aspect )
            print( "review pdf", review_pdf )
            print( i[0] )
    for z in neg_pairs:
        if (h == z[1] and c_neg < 3):
            review_aspect = []
            if reviews[z[0]] not in review_pdf:
                review_aspect.append( h )
                review_aspect.append( "Negative" )
                review_aspect.append( reviews[z[0]] )
                c_neg += 1
            print( "z[0]", z[0] )

            review_pdf.append( review_aspect )

for z in review_pdf:
    print( z )

import numpy as np
import matplotlib.pyplot as plt

# data to plot
aspects = aspects[: len( aspects ) - 1]
n_groups = len( aspects )

print( len( aspects ) )
print( len( counter_neg ) )

# create plot
fig, ax = plt.subplots()
index = np.arange( n_groups )
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar( index, counter_pos, bar_width,
                  alpha=opacity,
                  color='b',
                  label='pos' )

rects2 = plt.bar( index + bar_width, counter_neg, bar_width,
                  alpha=opacity,
                  color='g',
                  label='neg' )

plt.xlabel( 'Aspects' )
plt.ylabel( 'Scores' )
plt.title( 'Aspects Summary' )
plt.xticks( index + bar_width, aspects )
plt.legend()

plt.setp( ax.get_xticklabels(), rotation=30, horizontalalignment='right' )
plt.tight_layout()

plt.savefig( 'testplot.png', orientation='landscape' )
plt.show()

pdf = FPDF( 'P', 'mm', 'A4' )
pdf.set_fill_color( 16, 190, 188 )
pdf.set_draw_color( 16, 190, 188 )

pdf.add_page()
pdf.set_font( "Courier", size=12 )
image = "testplot.png"

pdf.multi_cell( 200, 20, txt="Review Analyzer Summary", align="C" )
for i in range( len( review_pdf ) ):
    pdf.multi_cell( 70, 7, txt=review_pdf[i][0] + " (" + review_pdf[i][1] + "): ", border=1, align="L", fill=1 )
    pdf.multi_cell( 200, 10, txt=review_pdf[i][2], align="L" )
pdf.add_page()
pdf.image( 'testplot.png', 0, 0, 200 )
pdf.output( "simple_demo.pdf" )