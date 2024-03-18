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

text = input('Wirte Your text hier :' )
text_input = tokenizer(text , return_tensors = 'tf')


input_ids = text_input['input_ids']

#prediction sentiment of input_ids
predictions = model.predict([input_ids])
logits = predictions.logits

import numpy as np

predicted_class = np.argmax(logits)

print(predicted_class) # output number 3 if the text example as  (i like my work ) 


# how bert work ?

sentiment_mapping = {
    0:'evry negative',
    1:'negative',
    2:'neutral',
    3:'positive',
    4:'very positive'
}

predicted_sentiment = sentiment_mapping[predicted_class]

print(predicted_sentiment)
