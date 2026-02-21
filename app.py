import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import nltk

nltk.download('punkt')

nltk.download('stopwords')
def transform(text):
    text = text.lower()
    text = text.split()
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf=pickle.load(open('vectorizor.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/ SMS Spam Classifier")
input_sms=st.text_input("Enter your message")
if st.button("Predict"):
# 1.preprocessing
    transform_text=transform(input_sms)
# 2.vectorize
    vector_input=tfidf.transform([transform_text])
# 3. predict
    result=model.predict(vector_input)[0]
    if result==1:
        st.text("SPAM")
    else:
        st.text("NOT SPAM")
