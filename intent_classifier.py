# -*- coding: utf-8 -*-

import json

from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PyDictionary import PyDictionary

class IntentClassifier:
    def __init__(self, classes ):
        self.file_path = ''
        self.classes = classes
        self.Tokenizer = 0
        self.model = 0
        
        
    def update_path (self, file_path):
        
        self.file_path = file_path

    def is_ready(self):
    
        if self.load():
            return True
        else:
            return False
             

    def load(self):
        try:
            self.model = load_model(self.file_path)
            return True
        
        except IOError:
            return False
        
        except ImportError:
            return False
        
        except Exception:
            return False
        
    def check_text (self, sentence ):
        """
        check_text (self, sentence ) checks if the input parameter 'sentence' is human readable (using PyDictionary)
        """
        dictionary=PyDictionary()
        
        words = sentence.split()

        for item in words:
            x = dictionary.meaning(item)
            if x==None:
                #print (item + ': Not a valid word')
                return False
            else:
                #print (item + ': Valid')
                return True
        
    
    def predict(self, sentence):
        """
        predict(self, sentence): takes the input as a string and returns top 3 intent classification results as a dictionary with 
        label as key and confidence as the value. 

        The label string is the intent label name.
        The confidence is the probability for the predicted intent.

        The output format is like: {{'label 1': highest confidence , 'label2': second highest confidence, 'label3': third highest confidence}}

        input example: "find me a flight that flies from memphis to tacoma"
        output example: {'flight': 1.0, 'airline': 0.0, 'flight_time': 0.0}
        """
        
        self.Tokenizer = Tokenizer()
        self.Tokenizer.fit_on_texts([sentence])
        tokens = self.Tokenizer.texts_to_sequences(texts=[sentence])
        tokens = pad_sequences(tokens, maxlen = 6000)
        prediction = self.model.predict(np.array(tokens))
        pred = np.argmax(prediction)
        # get index of the top 3 intent prediction
        index = np.argpartition(prediction[0], -3)[-3:] 

        # get probabilities of the top 3 intent prediction
        prob =np.round( prediction[0][index] , 2) 
     
        # get labels of the top 3 intent prediction
        labels = [ self.classes[i] for i in index ]
       
        response = dict(zip(labels, prob))
        sorted_response = dict(sorted(response.items(), key = lambda x: x[1], reverse = True))
        
        
        
        return (sorted_response)
        


#if __name__ == '__main__':
 #   pass
