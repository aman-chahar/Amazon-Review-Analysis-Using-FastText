# Amazon-Review-Analysis-Using-FastText
**LinkedIn Post: FastText Sentiment Analysis on Amazon Customer Reviews**

Excited to share my recent project on sentiment analysis using FastText, focusing on a vast dataset of Amazon customer reviews. 🚀 The goal was to leverage machine learning to understand and predict sentiment based on user-generated content and star ratings. 🌟

**Dataset and Setup:**
I began by accessing the dataset from Kaggle, containing millions of Amazon reviews and their corresponding star ratings. The dataset was compressed, so I used FastText to efficiently handle large amounts of text data.

```python
# Creating path for Kaggle file
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

# Importing Amazon reviews Sentiment dataset
!kaggle datasets download -d bittlingmayer/amazonreviews

# Extracting a compressed dataset
from zipfile import ZipFile
dataset = '/content/amazonreviews.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
```

**Data Preparation:**
The dataset was initially in a compressed format, so I utilized Python's `bz2` library to decompress it. Subsequently, I converted the data into a CSV format for easier handling.

```python
# Importing required libraries
import pandas as pd
import bz2
import csv

input_file_path = "/content/train.ft.txt.bz2"
output_file_path = "/content/train.csv"

# Decompressing the data and converting to CSV
with bz2.BZ2File(input_file_path, 'rb') as f:
    decompressed_data = f.read().decode('utf-8')

with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    for line in decompressed_data.splitlines():
        label, text = line.split(' ', 1)
        label = label.replace('__label__', '')  # Remove '__label__' prefix
        csv_writer.writerow([label, text])

print("Conversion completed. CSV file saved at:", output_file_path)
```

**Data Processing and Model Building:**
I then loaded the data into a Pandas DataFrame, performed basic data exploration, and split it into training and testing sets. For text processing, I used Keras to tokenize and pad sequences, preparing the data for the LSTM (Long Short-Term Memory) model.

```python
# Loading and processing data
data = pd.read_csv('/content/train.csv', names=["label", "text"])
data = data.dropna()

# Splitting data
xtrain, xtest, ytrain, ytest = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# Tokenization and padding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 5000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(xtrain)
X_train_seq = tokenizer.texts_to_sequences(xtrain)
X_test_seq = tokenizer.texts_to_sequences(xtest)
maxlen = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Building and training the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from sklearn.metrics import classification_report

embedding_dim = 100
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_pad, ytrain, epochs=8, batch_size=128, validation_split=0.2)
```

**Results and Conclusion:**
The LSTM model achieved impressive accuracy in predicting sentiment from Amazon reviews. Excited to delve deeper into the insights derived from customer sentiments and explore applications in enhancing user experience and decision-making! Stay tuned for more updates! 📊🔍 #SentimentAnalysis #FastText #MachineLearning #DataScience

Feel free to engage and share your thoughts! 😊
