import pickle
import re
import warnings

import gensim
import keras
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
# from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

# POS tagger dictionary
# pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

# show all columns
pd.set_option('display.max_columns', None)


def clean_text(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = ' '.join([word for word in text.strip().split() if word not in stopwords.words('english')])

    print(text, end="\t")

    return text


# def token_stop_pos(text):
#     tags = pos_tag(word_tokenize(text))
#     newlist = []
#     for word, tag in tags:
#         if word.lower() not in set(stopwords.words('english')):
#             newlist.append(tuple([word, pos_dict.get(tag[0])]))
#     return newlist


def lemmatize(pos_data):
    print(pos_data)
    print(type(pos_data)) # <class 'str'>
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

def token_stop_pos(text):
    # Tokenize the text into individual words
    words = nltk.word_tokenize(text)

    # Apply POS tagging to the words
    tagged_words = nltk.pos_tag(words)

    # Join the tagged words back into a single string
    return ' '.join([f'{word}/{tag}' for word, tag in tagged_words])

def process_data_sentiment(df):
    # load csv
    # df = pd.read_csv("dataset/Food_ordering_feedback_dataset.csv")

    # drop null values
    df = df.dropna()
    # remove duplicates
    df = df.drop_duplicates()

    print("Cleaning text")
    # for every 100 rows save the csv while running df['Text'].apply(lambda x: clean_text(x.lower()))
    batch_size = 100
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    for i, batch in enumerate(batches):
        batch['cleaned_text'] = batch['Text'].apply(lambda x: clean_text(x.lower()))
        print(batch.tail())
        # print columns
        print(batch.columns)
        batch['POS tagged'] = batch['cleaned_text'].apply(token_stop_pos)
        print(batch.columns)
        print(batch.tail())
        print("POS tagging text...")
        # send batch to lemmatize function
        batch['Lemma'] = batch['POS tagged']
        # batch['Lemma'] = batch['POS tagged'].apply(lemmatize)
        # batch['Lemma'] = batch['POS tagged'].apply(lambda x: lemmatize(x))
        print(batch.columns)
        batch.to_csv(f'data/sentiment/clean/batch_{i}.csv', index=False)
        df[i * batch_size:i * batch_size + len(batch)] = batch

    print(df.tail())

    a = input("Do you need to continue? (y/n)")
    if a == 'n':
        return

    # print("Tokenizing text")
    # df['tokenized_text'] = df['cleaned_text'].apply(lambda x: word_tokenize(x))
    # batch_size = 100
    # batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    # for i, batch in enumerate(batches):
    #     batch['Text'] = batch['Text'].apply(lambda x: clean_text(x.lower()))
    #     batch.to_csv(f'tokanize/batch_{i}.csv', index=False)
    #
    # print("POS tagging text")
    # df['POS tagged'] = df['cleaned_text']\
    #     # .apply(token_stop_pos)
    # print("Lemmatizing text")
    # df['Lemma'] = df['POS tagged'].apply(lemmatize)
    # to csv
    df.to_csv("data/sentiment/lemma.csv", index=False)


def clean_chars(data):
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    return data


def sentence_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def train_sentiment():
    # load csv
    # df = pd.read_csv("data/sentiment/lemma.csv")
    df = pd.read_csv("data/sentiment/Reviews.csv")
    print("Data loaded")
    process_data_sentiment(df)
    print("Data processed")

    # drop null values
    df = df.dropna()
    # remove duplicates
    df = df.drop_duplicates()


    # keep only column_name, value
    df = df[['column_name', 'value']]

    # rename to selected_text	sentiment
    df = df.rename(columns={'column_name': 'sentiment', 'value': 'selected_text'})

    print(df.tail())

    df["selected_text"].isnull().sum()
    df["selected_text"].fillna("No content", inplace=True)

    temp = []
    # Splitting pd.Series to list
    data_to_list = df['selected_text'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(clean_chars(data_to_list[i]))
    list(temp[:5])

    data_words = list(sentence_to_words(temp))

    data = []
    for i in range(len(data_words)):
        data.append(TreebankWordDetokenizer().detokenize(data_words[i]))
    print(data[:5])

    data = np.array(data)
    print(data[:5])

    # Label encoding
    labels = np.array(df['sentiment'])
    y = []
    for i in range(len(labels)):
        if labels[i] == 1:
            y.append(0)
        if labels[i] == 2:
            y.append(0)
        if labels[i] == 3:
            y.append(1)
        if labels[i] == 4:
            y.append(2)
        if labels[i] == 5:
            y.append(2)
    y = np.array(y)
    labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
    del y

    # Data sequencing and splitting
    max_words = 5000
    max_len = 200

    print("Tokenizing data...")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    features = pad_sequences(sequences, maxlen=max_len)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # save to npy
    np.save('data/sentiment/X_train.npy', X_train)
    np.save('data/sentiment/X_test.npy', X_test)
    np.save('data/sentiment/y_train.npy', y_train)
    np.save('data/sentiment/y_test.npy', y_test)

    # save tokenizer
    with open('data/sentiment/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # model
    model1 = Sequential()
    model1.add(layers.Embedding(max_words, 20))
    model1.add(layers.LSTM(20, dropout=0.5))
    model1.add(layers.Dense(3, activation='softmax'))

    print("Compiling model...")
    model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint1 = ModelCheckpoint("data/sentiment/best_model.hdf5",
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1,
                                  save_weights_only=False)
    early = EarlyStopping(monitor="val_accuracy", mode="max", patience=20)
    model1.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint1, early])

    # save the model
    model1.save('data/sentiment/model.h5')

    # save model weights
    model1.save_weights('data/sentiment/model_weights.h5')

    print("Model saved")


def predict_sentiment(text):
    # load tokenizer
    with open('data/sentiment/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Let's load the best model obtained during training
    best_model = keras.models.load_model("data/sentiment/best_model.hdf5")
    best_model.summary()

    max_len = 200
    sentiment = ['Neutral', 'Negative', 'Positive']

    sequence = tokenizer.texts_to_sequences([text])
    test = pad_sequences(sequence, maxlen=max_len)
    print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])

    return sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]


def pre_process(df):
    # drop null values
    df = df.dropna()
    # drop duplicates
    df = df.drop_duplicates()

    # keep only column_name, value
    df = df[['column_name', 'value']]
    # rename to selected_text	sentiment
    df = df.rename(columns={'column_name': 'sentiment', 'value': 'selected_text'})

    df["selected_text"].isnull().sum()
    df["selected_text"].fillna("No content", inplace=True)

    return df


# train_sentiment()