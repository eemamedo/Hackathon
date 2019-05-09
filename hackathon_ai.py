#https://www.youtube.com/watch?v=3dHu-xOpljU
'''
Monitor a folder for changes and notify what files have been changed

Useful Links:
    http://timgolden.me.uk/python/win32_how_do_i/watch_directory_for_changes.html
'''

'''
Classification
'''

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

print (twenty_train.data[0])

'''
After transforming the text into a bag of 
words, we can calculate various measures
to characterize the document. In order to do
so we have to generate a vector for each document that represents
the number of times each entry in the bag of words appears in the text.

Each entry of the lists refers to frequency or count of the corresponding entry in the bag-of-words
list. When we have a stacked collection of row vectors, where each row corresponds to a document (vector)
and each column corresponds to a word in the bag-of-words list, then this will be known as our term-frequency
document matrix. 
'''

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
X_train_tf = count_vect.fit_transform(twenty_train.data)
X_train_tf.shape #this is a term-frequency matrix
'''
Term-frequency matrix is a sparse matrix where each row is a document in out training corpus (D) and each column
corresponds to a term/word in the bag-of-words list
However, there are many common words like "the", "a", "to" etc.
To address this issue, we need to construct an inverse document frequency (idf).
This results in the term frequency-inverse document frequency (tf-idf) matrix.
IDF measure of how much information the word provides, that is, whether the term is common or rare across all 
documents in the corpus.

TF-IDF matrix is a sparse matrix, where each row is a document in our training corpus and each column corresponds
to a word in the bag-of-words list.
'''

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer() #check the docs what are the args
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape

#using stopwords (removing them) will improve result

#path = "/Users/Emin/Desktop/hackathon/vector_space_modelling-master/Code/Data"
data = open("/Users/Emin/Desktop/hackathon/vector_space_modelling-master/Code/Data/trainingdata.txt","w+")
print (data)

def get_tag_and_training_data(filename):
    '''takes the input file and returns  tokenized sentences and document tags as separate lists'''
    tags=[]
    documents=[]
    line_counter=1
    with open(filename) as f:
        for line in f:
            #skip first line
            if line_counter==1:
                line_counter=line_counter+1
                continue
            #Initialize the token list for line
            tags.append(int(line[:1]))
            documents.append(line[2:])
    return tags,documents

Y,X=get_tag_and_training_data('/Users/Emin/Desktop/hackathon/vector_space_modelling-master/Code/Data/trainingdata.txt')