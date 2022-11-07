import os
import pandas as pd
import string
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords


'''
In the following section we define the variables needed for the functions, for example the stopwords we want to remove
    or the punctuation we replace
'''
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation.replace('!', '').replace('?', '')  # ! and ? are not removed, they may indicate a sentiment
EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley"
}

'''
I decided to use five methods for the preprocessing: lowercasing and removing punctuation, stopwords, urls and 
    emoticons, an explanation why can be found in the report.
'''
def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCTUATION))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


'''
In the preprocess() function all the methods implemented above are called on a review and the resulting text is stored 
    in a dataframe 
'''
def preprocess(text, df, label):
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_emoticons(text)
    text = remove_urls(text)
    df.loc[len(df.index)] = [text, label]


'''
Creating an empty dataframe for train and test data. We only need two columns: the text of the review and the class
    (if the review is positive or negative)
We also declare the names of the folders the data is in, this makes it easier to iterate through them in the code below
'''
train_df = pd.DataFrame(columns=['review', 'label'])
test_df = pd.DataFrame(columns=['review', 'label'])
all_df = [train_df, test_df]
folders = ['train', 'test']
subfolders = ['neg', 'pos']

'''
The following part just visualizes how the function works by using the first text in our folders, in each step it 
    performs one of the methods and prints the result  
'''
path = 'aclImdb/train/neg'
for file in os.listdir(path)[:1]:
    review = open('aclImdb/train/neg/' + str(file), encoding='utf-8').read()

    print(f'\n-------- new review -------- \n---initial text: \n{review}')
    review = lowercase(review)
    print(f'---text after lowercasing: \n{review}')
    review = remove_punctuation(review)
    print(f'---text after removing punctuation (except ! and ?): \n{review}')
    review = remove_stopwords(review)
    print(f'---text after removing stopwords: \n{review}')
    review = remove_emoticons(review)
    review = remove_urls(review)
    print(f'---text after removing URLs and emoticons: \n{review}')

'''
This for-loop iterates through each file in each folder, calls the preprocess function and stores it with the folder's
    label to the right dataframe. Label 0 stands for a negative and label 1 for a positive review.
'''
for f in range(len(folders)):
    for sf in range(len(subfolders)):
        path = 'aclImdb/' + folders[f] + '/' + subfolders[sf]
        for file in os.listdir(path):
            review = open(path + '/' + str(file), encoding='utf-8').read()
            preprocess(review, all_df[f], str(sf))

'''
Finally we get our two final dataframes, they each contain 25.000 rows with a preprocessed review and a label
Label 0: negative review
Label 1: positive review
'''
print('Training dataframe:\n', train_df, '\n\n')
print('Test dataframe\n', test_df)
