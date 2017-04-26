import string
import pprint
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus.europarl_raw import english

def main():
    documents = ['Hello, how are you!',
                 'Win money, win from home.',
                 'Call me now.',
                 'Hello, Call hello you tomorrow?']
    
    lower_case_documents = []
    for i in documents:
        lower_case_documents.append(i.lower())

    preprocessed_documents = []    
    for i in lower_case_documents:
        preprocessed_documents.append(i.translate(string.maketrans('', ''), string.punctuation).split())
    print(preprocessed_documents)

    frequency_list = []    
    for i in preprocessed_documents:
        frequency_counts = Counter(i)
        frequency_list.append(frequency_counts)
    pprint.pprint(frequency_list, indent=7)
    
    # end of our own scikit-learn CountVectorizer. Now we turn to already implemented version in scikit-learn

    count_vector = CountVectorizer(stop_words=None)
    count_vector.fit(documents)
    doc_array = count_vector.transform(documents).toarray()
    pprint.pprint(doc_array)
    
    frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
    pprint.pprint(frequency_matrix)
    
    

if __name__ == "__main__": main()