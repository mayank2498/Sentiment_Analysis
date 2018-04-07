import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('Positive_reviews').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('Negative_reviews').read())
negative_reviews = negative_reviews.findAll('review_text')

#t = positive_reviews[0]
#nltk.tokenize.word_tokenize(t.text)

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


print(my_tokenizer(t.text))


word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []


for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

            
# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    #print("\nSum is ",x.sum())
    x = x / x.sum() # normalize it before setting label
    #test = np.array([a for a in x if a > 0])
    #print(test)
    #print("\n\n")
    #print(test.sum())
    x[-1] = label
    return x


#new = tokens_to_vector(negative_tokenized[0],1)


N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))


i = 0 
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

    
X = data[:,:-1]
y = data[:,-1]    

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification rate:", model.score(X_test, y_test))


threshold = 0.5
for word,index in word_index_map.items():
    weight = model.coef_[0][index]
    if(weight > threshold or weight< -threshold ):
        print("\n",word," : ",weight)
