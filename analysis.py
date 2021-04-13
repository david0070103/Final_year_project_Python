import spacy
import pandas as pd
import numpy as np
import string
from textblob import TextBlob

class analyser:
    def __init__(self):
        self.en_nlp = spacy.load("en_core_web_sm")
        self.question_classification = pd.DataFrame(columns=['question_word','question_pos', 'question_follow_word', 'question_follow_word_pos',
                                               'main_verb', 'main_verb_pos', 'keyword_list', 'objective'])

    def process_question(self, question, qclass, en_nlp, count):
        en_doc = en_nlp(question)
        process_list = list(en_doc.sents)
        process = process_list[0]
        root_token = ""
        root_text = ""
        
        wh_pos = ""
        wh_word = ""
        
        wh_ne_gram = ""
        wh_ne_pos = ""
        
        nn_list = ''
        objective = ''
        
        for token in process:
            if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
                wh_pos = token.tag_
                wh_word = token.text
                wh_ne_gram = str(en_doc[token.i + 1])
                temp = TextBlob(str(en_doc[token.i + 1])).tags
                if temp:
                    wh_ne_pos = temp[0][1]
            elif token.dep_ == "ROOT":
                root_token = token.tag_
                root_text = token.text
            elif token.tag_ == "NN" or token.tag_ == "NNS" or token.tag_ == "NNP" or token.tag_ == "NNPS":
                if (token.tag_ == "NN") & (token.dep_ == "nsubj"):
                    objective = token.text
                    continue
                if (token.dep_ == 'compound'): # There are compound noun (Example: Tuen Mun)
                    nn_list = nn_list + token.text + ' '
                    continue
                nn_list = nn_list + token.text + ','
            
            else:
                continue
        if (nn_list != ''):
            if(nn_list[-1] == ','):
                nn_list = nn_list[:-1]
        self.question_classification.loc[count] = [wh_word.lower(), wh_pos, wh_ne_gram.lower(), wh_ne_pos, root_text, root_token,nn_list, objective]

    def put_question(self, question):
        count = 0
        question
        for i in question:
            self.process_question(i,'',self.en_nlp, count)
            count+=1
        return self.question_classification.iloc[0]['keyword_list']