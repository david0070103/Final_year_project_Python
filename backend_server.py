import pickle
import pandas as pd
import sklearn
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc

import analysis
import nltk
import pymysql
import string
import requests

from flask import jsonify

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
    def questionHandler(self, question):
        question = question.split(' ')
        actual_question = ''

        # Ignore and count some wording which are useful but may affect the result
        for i in question:
            if(i.lower() in self.ignore_list_price):
                self.ignore_list_obj_price[i.lower()] += 1
            else:
                actual_question += i+' '
        if(len(actual_question.split(' ')) <= 1):
            return -1
        actual_question = [actual_question]
        print('Question >>>', actual_question)
        # analysis question keyword
        keyword = self.analysis.put_question(actual_question)
        print('Keyword >>>', keyword)

        # count the perposition from, to, in, at
        index_from = 0
        index_to = 0
        count = 0

        # Handle the case with differnet sentence structure, (from/ in/ at) ... to ... / to ... (from/ in/ at)...
        for i in question:
            if (i.lower() == 'from') | (i.lower() == 'in') | (i.lower() == 'at') | (i.lower() == 'between'):
                index_from = count
            if (i.lower() == 'to') | (i.lower() == 'and'):
                index_to = count
            count += 1

        keyword = keyword.lower().split(',')
        if (index_from > index_to) & (len(keyword) == 2):
            keyword = [keyword[1], keyword[0]]

        # Return lowercase format [keyword0 ,keyword1] (keywords !)
        return keyword

    # Start of handlers

    def priceHandler(self, keyword):
        if (len(keyword) != 2 ):
            return self.generate_respond(-1, ['Station name not found / Wrong input for asking fare'], [])
        resp =  self.knowledgeBased_Retrieval_Price(keyword[0], keyword[1])

        if resp != []:
            return self.generate_respond(1, resp, [
                'What is the route from {} to {} ?'.format(string.capwords(keyword[0]), string.capwords(keyword[1])),
                'What is the nearest exit from {} to (destination) ?'.format(string.capwords(keyword[1]))
                ])
        else: 
            return self.generate_respond(-1,['Invalid station name, please input again.'], [])
    
    def toiletHandler(self, keyword):
        resp = self.knowledgeBased_Retrieval_Toilet(keyword)
        return self.generate_respond(1, resp, [])

    def locationHandler(self, keyword, question, sub_type):
        start_index = 0
        dest_index = 0
        
        # Because there are only two place exist if asking about the starting point and destination / nearest exit from station and place
        if(len(keyword) != 2):
            resp = self.documentation_retrieval(question, sub_type)
            return self.generate_respond(1, resp, [])

        # Check there are any wrong input with the station name
        keyword = self.checkStationName([keyword[0], keyword[1]])

        # Finding out the index of starting point and destination
        for key, value in self.station_name.items():
            if value == keyword[0]:
                start_index = key
            if value == keyword[1]:
                dest_index = key

        print('Start index >>>', start_index)
        print('Dest_index >>>', dest_index)
        # If there is only one station exist and there is an another destination point exist
        if (start_index == 0) | (dest_index == 0):
            db = self.knowledgeBased_Retrieval_Location(keyword[0], keyword[1])
            # Documentation retrieval used for two place but without asking the route / nearest exit of station
            print('DB >>>', db)
            if(db == []):
                answer = self.documentation_retrieval(question, sub_type)
                return self.generate_respond(1, answer, [])
            # Output the database result if database query result
            return self.generate_respond(1, db, [])
        
        return self.apiHanlder(start_index, dest_index)
    
    def apiHanlder(self, start_index, dest_index):
        # API part
        resp = requests.get('http://www.mtr.com.hk/share/customer/jp/api/HRRoutes/?lang=E', params={'o': start_index, 'd' : dest_index})
        resp = resp.json()
        route = resp['routes'][0]['path']
        walk_interchange = 0
        output =[]
        output.append('Starting point: '+ string.capwords(self.station_name.get(start_index)))
        for i in route:
            if (i['linkType'] == 'INTERCHANGE'):
                if walk_interchange == 0:
                    output.append('Travel to ' + string.capwords(self.station_name.get(i['ID'])))
                walk_interchange = 0
            if (i['linkType'] == 'WALKINTERCHANGE'):
                output.append('Travel to ' + string.capwords(self.station_name.get(i['ID'])))
                walk_interchange = 1
            if i['linkText'] != None:
                output.append(i['linkText'])
            if i['linkType'] == 'END':
                output.append('Destination station: ' + string.capwords(self.station_name.get(i['ID'])))
        output.append('Estimatede time: ' + str(resp['routes'][0]['time']) + 'mins')
        return self.generate_respond(1, output, [
            'What is the price from {} to {} ?'.format(string.capwords(self.station_name.get(start_index)),string.capwords(self.station_name.get(dest_index))),
            'What is the nearest exit from {} to (destination) ?'.format(string.capwords(self.station_name.get(dest_index)))])

    # End of handlers

    # Start of knowledge based retrieval

    def knowledgeBased_Retrieval_Price(self, keyword0, keyword1):
        check_keyword = self.checkStationName([keyword0, keyword1])

        # Running SQL to getting price
        self.cursor.execute('SELECT * FROM travel_fee WHERE src_station_name = %s AND dest_station_name = %s', [check_keyword[0], check_keyword[1]])
        temp = []
        # Print the result
        for r in self.cursor:
            if(self.ignore_list_obj_price['adult'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                temp.append('Adult octopus: $' + str(r[5]))
                return temp
            elif(self.ignore_list_obj_price['student'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                temp.append('Student octopus: $' + str(r[6]))
                return temp
            elif(self.ignore_list_obj_price['adult'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                temp.append('Adult ticket: $'+ str(r[7]))
                return temp
            elif(self.ignore_list_obj_price['children'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                temp.append('Children octopus: $'+ str(r[8]))
                return temp
            elif(self.ignore_list_obj_price['elderly'] == 1 & self.ignore_list_obj_price['octopus'] == 1):
                temp.append('Elderly octopus: $'+ str(r[9]))
                return temp
            elif(self.ignore_list_obj_price['children'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                temp.append('Children ticket: $'+ str(r[11]))
                return temp
            elif(self.ignore_list_obj_price['elderly'] == 1 & self.ignore_list_obj_price['ticket'] == 1):
                temp.append('Eldery ticket: $'+ str(r[12]))
                return temp
            else:
                temp.append('Adult octopus: $'+ str(r[5]))
                temp.append('Student octopus: $'+ str(r[6]))
                temp.append('Adult ticket: $'+str(r[7]))
                temp.append('Children octopus: $'+str(r[8]))
                temp.append('Elderly octopus: $'+ str(r[9]))
                temp.append('Children ticket: $'+ str(r[11]))
                temp.append('Elderly ticket: $'+ str(r[12]))
                return temp
        return []

    # Case for asking the toilet in the station
    def knowledgeBased_Retrieval_Toilet(self, keyword):
        toilet_flag = 0
        resp = []
        for i in keyword:
            if (i in self.station_name.values()):
                toilet_flag = 1
                sql = 'SELECT * FROM mtr_toilet WHERE station = %s ORDER BY inside DESC'
                self.cursor.execute(sql, [i])
                if (self.cursor.rowcount > 0):
                    resp.append('Available toilets in ' + string.capwords(i)+ ' :')
                    for record in self.cursor:
                        if(record[2] == 'yes'):
                            resp.append('At station / public toliets: ' + record[3])
                        else:
                            resp.append('Shopping mall or private property: ' + record[3])
                    return resp
                else:
                    resp.append('No toilet in this MTR station')
                    return resp
        if toilet_flag == 0:
            resp.append('There is no such station in MTR list.')
            return resp
        return

    # Case for asking the nearest exit
    def knowledgeBased_Retrieval_Location(self, keyword0, keyword1):
        resp = []
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

        if self.cursor.rowcount > 0:
            for r in self.cursor:
                resp.append('The nearest exit is Exit '+ string.capwords(r[2]))
        return resp

    # End of knowledge based retrieval


    def documentation_retrieval(self, questions, ques_sub_type):
        sub_type = ['code', 'count', 'money', 'other', 'desc', 'def']
        if ques_sub_type in sub_type:
            fileContent = ''
            with open("./document/document_text.txt","r") as file:
                fileContent = file.read()
            predict = self.predictor.predict(
                passage= fileContent,
                question= questions
            )
            return [predict['best_span_str']]
        else:
            return []

    def generate_respond(self, status, message, prediction):
        if(status == 1):
            return jsonify({
                'status' : 1,
                'result' : message,
                'predict': prediction
            })
        else:
            return jsonify({
                'status' : -1,
                'dataMessage': message
            })