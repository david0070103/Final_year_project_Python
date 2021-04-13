import pickle
import pandas as pd
import sklearn
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc

import analysis
import nltk
import pymysql
import string

class SVM_Linear():
    def __init__(self):
        super(SVM_Linear, self).__init__()
        self.types = {
        "ENTY": "entity",
        "DESC": "description",
        "LOC": "location",
        "NUM": "numeric"
        }

        # Ignore List initialize
        self.ignore_list_obj_price = {
            'elderly': 0,
            'adult': 0,
            'children': 0,
            'student': 0,
            'octopus': 0,
            'ticket': 0,
            'station': 0,
            'exit' : 0,
            'distance': 0,
            'path': 0,
            'toilet': 0,
            'toilets': 0,
        }
        self.ignore_list_price = ['elderly', 'adult', 'children', 'octopus', 'ticket', 'station', 'exit', 'distance', 'path', 'toilet', 'toilets']

        # Get models and package that stored specific data
        self.vectorizer = pickle.load(open('./pickle/vectorizer.tfid', 'rb'))
        self.svm_linear = pickle.load(open('./pickle/svm_linear.model', 'rb'))
        self.type_dict = pickle.load(open('./pickle/type.dict', 'rb'))
        self.station_name = pickle.load(open('./pickle/station.name', 'rb'))

        # Features = columns of model used to transform the data
        self.features = self.vectorizer.get_feature_names()

        # Database Management
        self.mydb =  pymysql.connect(
            host="localhost",
            user="root",
            password="12345678",
            database="final_year_project"
        )
        self.cursor = self.mydb.cursor()

        # Analyze the question keyword
        self.analysis = analysis.analyser()

        # Allennlp
        self.predictor = Predictor.from_path("./bidaf-elmo-model-2020.03.19.tar.gz")
    
    def init(self):
        self.ignore_list_obj_price = {
            'elderly': 0,
            'adult': 0,
            'children': 0,
            'student': 0,
            'octopus': 0,
            'ticket': 0,
            'station': 0,
            'exit' : 0,
            'distance': 0,
            'path': 0,
            'toilet': 0,
            'toilets': 0,
        }

    def transfer_type(self, question):
        # Initialize
        self.init()

        temp_list = []
        temp_list.append(question)
        
        # Transform the question into the format that the model can read
        question= self.vectorizer.transform(temp_list)
        del temp_list
        question = question.todense().tolist()
        question_df = pd.DataFrame(question, columns= self.features)
        del question

        # Predict the type of the question
        res = self.svm_linear.predict(question_df)
        del question_df

        # Split the main and sub type of the question
        res = self.type_dict[res[0]].split('_')
        main_type = [self.types[res[0]], res[1]]
        del res

        return main_type


    def checkStationName(self, stationList):
        tempList = []
        
        # check the station in keywordList
        for station in stationList:
            for i in self.station_name.values():
                # check the difference between the keyword and origin station name
                dis = nltk.edit_distance(station, i)
                if (dis == 1):
                    station = i
            tempList.append(station)
        return tempList

    # Function that used to get back the keywords
    def questionHandler(self, question, res0, res1):
        question = question.split(' ')
        actual_question = ''

        # Ignore and count some wording which are useful but may affect the result
        for i in question:
            if(i.lower() in self.ignore_list_price):
                self.ignore_list_obj_price[i.lower()] += 1
            else:
                actual_question += i+' '
        actual_question = [actual_question]

        print('Question >>>', actual_question)
        # analysis question keyword
        keyword = self.analysis.put_question(actual_question)

        # count the 
        index_from = 0
        index_to = 0
        count = 0

        # Handle the case with differnet sentence structure, from ... to ... / to ... from ...
        for i in question:
            if (i == 'from') | (i == 'From') | (i == 'In') | (i == 'in'):
                index_from = count
            if (i == 'to') | (i == 'To'):
                index_to = count
            count += 1
        keyword = keyword.lower().split(',')
        if (index_from > index_to) & (len(keyword) == 2):
            keyword = [keyword[1], keyword[0]]

        # Return lowercase format [keyword0 ,keyword1]
        return keyword

    def knowledgeBased_Retrieval_Price(self, keyword0, keyword1):
        check_keyword = self.checkStationName([keyword0, keyword1])

        # Running SQL to getting price
        self.cursor.execute('SELECT * FROM travel_fee WHERE src_station_name = %s AND dest_station_name = %s', [check_keyword[0], check_keyword[1]])
        # Print the result
        flag = 0
        for r in self.cursor:
            flag = 1
            if(self.ignore_list_obj_price['adult'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                print('Adult octopus: $',r[5])
            elif(self.ignore_list_obj_price['student'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                print('Student octopus: $',r[6])
            elif(self.ignore_list_obj_price['adult'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                print('Adult ticket: $',r[7])
            elif(self.ignore_list_obj_price['children'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                print('Children octopus: $',r[8])
            elif(self.ignore_list_obj_price['elderly'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                print('Elderly octopus: $',r[9])
            elif(self.ignore_list_obj_price['children'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                print('Children ticket: $',r[11])
            elif(self.ignore_list_obj_price['elderly'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                print('Eldery ticket: $',r[12])
            else:
                print('Adult octopus: $', r[5])
                print('Student octopus: $', r[6])
                print('Adult ticket: $', r[7])
                print('Children octopus: $', r[8])
                print('Elderly octopus: $', r[9])
                print('Children ticket: $', r[11])
                print('Elderly ticket: $', r[12])
        if flag == 0:
            print('Invalid station name, please input again.')
        return


    def knowledgeBased_Retrieval_Location(self, keyword0, keyword1):

        # Key_flag = 0 (keyword0 , keyword1), key_flag = 1 (keyword1, keyword0) key_flag = 2 (Need to do documentation retrieval)
        key_flag = -1
        check_keyword = self.checkStationName([keyword0, keyword1])
        if check_keyword[0] in self.station_name.values():
            key_flag = 0
        elif check_keyword[1] in self.station_name.values():
            key_flag = 1

        # Querying the nearest exit
        if key_flag == 0:
            self.cursor.execute('SELECT * FROM nearest_exit WHERE station = %s AND venue = %s LIMIT 1', [check_keyword[0], check_keyword[1]])
        elif key_flag == 1:
            self.cursor.execute('SELECT * FROM nearest_exit WHERE station = %s AND venue = %s LIMIT 1', [check_keyword[1], check_keyword[0]])

        if self.cursor.rowcount != -1:
            for r in self.cursor:
                print('The nearest exit is Exit', r[2])
                key_flag = -2
        return key_flag


    def knowledgeBased_Retrieval_Toilet(self, keyword):
        toilet_flag = 0
        for i in keyword:
            if (i in self.station_name.values()):
                toilet_flag = 1
                sql = 'SELECT * FROM mtr_toilet WHERE station = %s ORDER BY inside DESC'
                self.cursor.execute(sql, [i])
                print('self.cursor', self.cursor.rowcount)
                if (self.cursor.rowcount > 0):
                    print('Available toilets in', string.capwords(i),':')
                    for record in self.cursor:
                        if(record[2] == 'yes'):
                            print('At station / public toliets:', record[3])
                        else:
                            print('Shopping mall or private property:', record[3])
                else:
                    print('No toilet in this MTR station')
        if toilet_flag == 0:
            print('There is no such station in MTR list.')
        return


    def documentation_retrieval(self, questions):
        fileContent = ''
        with open("./document/document_text.txt","r") as file:
            fileContent = file.read()
        predict = self.predictor.predict(
            passage= fileContent,
            question= questions
        )
        return predict['best_span_str']
