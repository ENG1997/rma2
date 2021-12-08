import streamlit as st
import nltk
import re
from string import punctuation
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from nltk.corpus import stopwords
df = pd.read_csv('prog_book.csv')
df.head()
nltk.download('stopwords')
stop = stopwords.words('english')
stop = set(stop)


def lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop])


def remove_digits(text):
    return re.sub(r'\d+', '', text)


def clean_text(text):
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_digits(text)
    return text


df['clean_Book_title'] = df['Book_title'].apply(clean_text)
df.head()

df['clean_Description'] = df['Description'].apply(clean_text)
df.head()

vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)

X = vectorizer.fit_transform(df['clean_Book_title'])
title_vectors = X.toarray()
desc_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
Y = desc_vectorizer.fit_transform(df['clean_Description'])
desc_vectors = Y.toarray()

def get_recommendations(value_of_element, feature_locate, df, vectors_array, feature_show):
    global simular
    index_of_element = df[df[feature_locate] == value_of_element].index.values[0]
    show_value_of_element = df.iloc[index_of_element][feature_show]
    df_without = df.drop(index_of_element).reset_index().drop(['index'], axis=1)
    vectors_array = list(vectors_array)
    target = vectors_array.pop(index_of_element).reshape(1, -1)
    vectors_array = np.array(vectors_array)
    most_similar_sklearn = cosine_similarity(target, vectors_array)[0]
    idx = (-most_similar_sklearn).argsort()
    all_values = df_without[[feature_show]]
    for _ in idx:
        simular = all_values.values[idx]
    recommendations_df = pd.DataFrame({feature_show: show_value_of_element,
                                       "First Book": simular[0][0],
                                       "Second Book": simular[1][0],
                                       "Third Book": simular[2][0]},
                                      index=[0])

    return recommendations_df



st.title("R-M-A Book Recommendation ")
df = pd.read_csv('prog_book.csv')
nltk.download('stopwords')
stop = stopwords.words('english')
stop = set(stop)

df['clean_Book_title'] = df['Book_title'].apply(clean_text)
df['clean_Description'] = df['Description'].apply(clean_text)

vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
X = vectorizer.fit_transform(df['clean_Book_title'])
title_vectors = X.toarray()

tk = 0

col1, col2 = st.columns(2)
books_dict = pd.read_csv('prog_book.csv')
books = pd.DataFrame(books_dict)
with col1:
    feat = st.sidebar.selectbox("Select Mode : ", ['Book_title', 'Rating', 'price'])
with col2:
    selected_book_name = st.sidebar.selectbox('Enter book name that you liked : ', books['Book_title'].values)
    if st.sidebar.button('RECOMMEND FOR ME'):
        tk = 1

if tk == 1:
    st.success('Recommending books similar to ' + selected_book_name)
    st.empty()
    st._arrow_dataframe((get_recommendations(selected_book_name, 'Book_title', df, title_vectors, feat)), width=1560, height= 2500)


with st.sidebar.expander("See explanation"):
    '\n'

    st.write("""when you select a book and click recommened for me 
    this Webapp will show you five books 
    like the book that you select  """)

with st.sidebar.expander("POWERED BY"):
    st.write(""" 
    LCT.RANA RYAD    \n                
    STUDENTS: \n
    MOHAMMED KHALID IBRAHIM \n
    ALI ANMAR BORHAN  \n
     """)
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'
'\n'

with st.expander("You Can See & Download Books Here"):
    j=0
    for i in books :
        while j<271 :
            st.image(books['image'].values[j], caption= books['Book_title'].values[j], width= 660)
            st.success(books['Description'].values[j])
            j +=1