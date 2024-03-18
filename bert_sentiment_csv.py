from transformers import BertTokenizer , TFBertForSequenceClassification

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'


# Spliter --->>
tokenizer = BertTokenizer.from_pretrained(model_name)

# model use name 
# nedded libraries :
# pip install tf-keras
#pip install TensorFlow
model = TFBertForSequenceClassification.from_pretrained(model_name)

# test the model 
import pandas as pd
csv_file = './IMDB Dataset.csv'
df = pd.read_csv(csv_file)

first_5_rows = df.head(5)

import numpy as np

sentiment_mapping = {
    0:'very negative',
    1:'negative',
    2:'neutral',
    3:'positive',
    4:'very positive'
}

first_5_rows_copy = first_5_rows.copy()
predicted_sentiments = []

for title in first_5_rows_copy['review']:
    text_input = tokenizer(title ,)

    
    input_ids = text_input['input_ids']
    predictions = model.predict([input_ids])
    logits = predictions.logits
    
    predicted_class = np.argmax(logits) #get max value of logits
    predicted_sentiment = sentiment_mapping[predicted_class] 
    predicted_sentiments.append(predicted_sentiment)
first_5_rows_copy['predicted_sentiment'] = predicted_sentiments


print(first_5_rows_copy)






