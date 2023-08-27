import streamlit as st
import keras
import tensorflow as tf
import requests
import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import pycountry
from keras.preprocessing.text import one_hot,Tokenizer
from keras.utils import pad_sequences

#Downloading some dependencies
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

#Loading the DL Model

model = tf.keras.models.load_model('News_Classification_bidirectional_lstm_model.keras')

#Setting the titles

st.markdown("<h1 style='text-align: center; color: white;'>Multi-Class News Classifier</h1>", unsafe_allow_html=True)

# # Creating a list of countries

# names_of_countries = []
# length = len(pycountry.countries)
# for i in range(length):
#     names_of_countries.append(list(pycountry.countries)[i].name)

#Fetching the country's ISO code
def fetch_country_code(name):
    code = pycountry.countries.get(name=name).alpha_2
    return code.lower()


#Fetching the news

def fetch_news(name):
    
    
    
    headers = {
       "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
}
    news = []

    global final
    # code = fetch_country_code(country_name)
    code = 'us'
    query = name
    
    url = 'https://newsapi.org/v2/everything?q={}&from=2023-08-25&to=2023-08-26&language=en&sortBy=popularity&apiKey=84c5e76001f14f0ca2651dd86ef80989'.format(query)
        
    data = requests.get(url, headers=headers).json()
    print('------------------------------------------------------------------------------------------------------------------')

    tot_res = data['totalResults']
    ls_name = []
    ls_title = []
    ls_desc = []
    ls_url = []
    ls_img_url = []
    publish_date = []
    ls_content = []
    print(url, tot_res)

    for i in tqdm(range(1, 6)):

            next_page_url = 'https://newsapi.org/v2/everything?q={}&language=en&page={}&sortBy=popularity&apiKey=84c5e76001f14f0ca2651dd86ef80989'.format(query, str(i+1))
            for j in data['articles']:

                    name = j['source']['name'] 
                    title = j['title']
                    desc = j['description']
                    url = j['url']
                    img_url = j['urlToImage']
                    date = j['publishedAt']
                    content = j['content']

                    ls_name.append(name)
                    ls_title.append(title)
                    ls_desc.append(desc)
                    ls_url.append(url)
                    ls_img_url.append(img_url)
                    ls_content.append(content)
                    publish_date.append(date)

            data = requests.get(next_page_url, headers=headers).json()
    
    dic = ({
                'name': ls_name,
                'title': ls_title,
                'description': ls_desc,
                'content': ls_content,
                'url': ls_url,
                'img_url': ls_img_url,
                'Date': publish_date
    })
        
    final = pd.DataFrame(dic)  
    final.to_csv('news.csv', index=False)
    
    
def preprocess():
    
    df = pd.read_csv('news.csv')
    print(df)

    
    df['tags'] = df['title'] + df['description'] + df['content']
    
    df['tags'] = df['tags'].astype('str')

    #Lowercasing

    df['tags'] =  df['tags'].str.lower()
    #Removing Contradictions

    import contractions

    def remove_contradictions(text):

        return " ".join([contractions.fix(word.text) for word in nlp(text)])

    df['tags']= df['tags'].apply(remove_contradictions)
    
    # Removing HTML tags

    import re

    def remove_html(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    df['tags'] =  df['tags'].apply(remove_html)

    #Remove @

    def remove_at_the_rate(text):

        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text != "@":
                new.append(word.text)

        return ' '.join(new)

    df['tags'] =  df['tags'].apply(remove_at_the_rate)
    
    #Removing URL

    import re

    def remove_url(text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)

    df['tags']=  df['tags'].apply(remove_url)


    #Remmove punctuation

    import string

    punc = string.punctuation

    def  remove_punc(text):

        return text.translate(str.maketrans('', '', punc))

    df['tags']=  df['tags'].apply(remove_punc)

    # Removing stop words


    from nltk.corpus import stopwords

    stopwords = stopwords.words('english')

    def remove_stop_words(text):
        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text not in stopwords:

                new.append(word.text)

        return ' '.join(new)

    df['tags'] =  df['tags'].apply(remove_stop_words)

    def Lemmetization(text):

        return " ".join([word.lemma_ for word in nlp(text)])


    df['tags'] =  df['tags'].apply(Lemmetization)
    
    def is_alpha(string):
    
        ls = string.split()
        new = []
        # print(ls)
        for word in ls:
            if word.isalpha()==True:
                new.append(word)
        return ' '.join(new)
    
    df['tags'] =  df['tags'].apply(is_alpha)
    
    df.to_csv('preprocessed.csv', index=False)

def predict():
    
    preprocessed = pd.read_csv('preprocessed.csv')
    tok = Tokenizer()
    tok.fit_on_texts(preprocessed['tags'])
    
    max_len = 200
    
    encd_news = tok.texts_to_sequences(preprocesssed['tags'])
    embd_dim = 200
    
    pad_news = pad_sequences(maxlen = max_len, padding='pre', sequences=encd_news)
    
    model.predict([pad_news], 1024)
    
name = st.text_input('Enter a keyword that you want your artciles to have!')
print(name)
fetch_news(name)
# print(final)
preprocess()

predict()