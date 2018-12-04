from aspect_extarction import AspectExtractor
import nltk
from nltk.corpus import product_reviews_1
import csv


class ABSA():
    with open( 'transaction_file.csv', mode='w', newline='' ) as transaction_file:
        transaction_writer = csv.writer( transaction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
        Aspect_Extractor = AspectExtractor()
        camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
        reviews = []
        Temp = []
        for i in range(len(camera_reviews)):
            review_nouns = []
            for j in range(len(camera_reviews[i].sents())):
                sent_nouns = []
                sentence = camera_reviews[i].sents()[j]
                sent_nouns = Aspect_Extractor.extract_nouns(sentence)
                print("NOUNS ABSA:", sent_nouns)
                review_nouns.append(sent_nouns)
                Temp.append(sent_nouns)
                transaction_writer.writerow(sent_nouns)
            reviews.append(review_nouns)
        Aspect_Extractor.extract_freq_nouns(Temp)




