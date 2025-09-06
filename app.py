import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM as KerasLSTM

from tensorflow.keras.preprocessing.text import Tokenizer
# class FixedLSTM(KerasLSTM):
#     def __init__(self, *args, **kwargs):
#         if 'time_major' in kwargs:
#             kwargs.pop('time_major')
#         super().__init__(*args, **kwargs)

model=load_model('my_model.h5')
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)[0]
    
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None
## Streamlit app
st.title("Next Word Prediction LSTM AND EARLY STOPPING")
input_text=st.text_input("Enter your text here:")
if st.button("Predict"):
    if input_text:
        max_sequence_len=model.input_shape[1]+1
        next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
        st.write(f"Next word: {next_word}")
    else:
        st.write("Please enter some text.")



