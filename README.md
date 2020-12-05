# 简介
本程序基于词袋模型和TF-IDF以及回归分类算法,主要来通过在IMDB电影评论数据集上应用简单的词袋模型，从而预测电影评论是积极的还是消极的。
## 读取数据
首先读取训练数据

```python
import pandas as pd       
train = pd.read_csv("F:\labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
test = pd.read_csv("F:/TMI/testData/testData.tsv", header=0, delimiter="\t",quoting=3 )

```

## 文本预处理
```python
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split() 
import nltk
from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english") )  
# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")] 

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(words )) 
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
clean_test_reviews = []
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 

print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
    clean_test_reviews.append( review_to_words( test["review"][i] ) )
print("Finished cleaning and parsing the training set movie reviews...\n")
```
## 从词袋中构造特征
```python
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
test_data_features=vectorizer.fit_transform(clean_test_reviews)
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
test_data_features=test_data_features.toarray()
print("Created finished\n")
```

## 构造回归模型并将其保存在model2.csv中
```python
print("Training the LogisticRegression...")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tf_w=TfidfVectorizer(max_features= 5000, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1,2),
            stop_words = 'english')
tf_w.fit(list(train['review'])+list(test['review']))
train_tfw=tf_w.transform(train['review'])
test_tfw=tf_w.transform(test['review'])
# Initialize a Random Forest classifier with 100 trees
LR=LogisticRegression()
LR.fit(train_tfw,train['sentiment'])
test_pred=LR.predict(test_tfw)
# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
output = pd.DataFrame( data={"id":test["id"], "sentiment":test_pred} )

# Use pandas to write the comma-separated output file
output.to_csv( "F:/TMI/testData/model2.csv", index=False, quoting=3 )
print("Training Finished.")
```

## 提交结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205195153730.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoMjAxOGk=,size_16,color_FFFFFF,t_70)
