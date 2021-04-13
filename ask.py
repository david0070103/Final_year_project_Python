import backend
import requests
import string

model = backend.SVM_Linear()

while True:
    question = str(input("Input question>> "))
    res = model.transfer_type(question)
    print('Type >>>', res)
    
    # keyword searching (for all cases)
    keyword = model.questionHandler(question, res[0], res[1])
    if(keyword == -1):
        print('Not a question, please reinput again !')
        continue

    # Case for getting price of the ticket
    if(res[1] == 'money'):
        if (len(keyword) != 2 ):
            print('Station name not found / Wrong input for asking fare')
            continue
        db = model.knowledgeBased_Retrieval_Price(keyword[0], keyword[1])


    # Case for asking toilet in MTR station
    elif (('wc' in [item.lower() for item in question.split(' ')])| ('toilet' in [item.lower() for item in question.split(' ')]) | ('toilets' in [item.lower() for item in question.split(' ')])):
        model.knowledgeBased_Retrieval_Toilet(keyword)


    # Case for asking nearest exit of MTR place
    elif (res[0] == 'location') & (res[1] == 'other'):
        # If there are keywords != 2 (Not asking for the exit)
        if(len(keyword) != 2):
            answer = model.documentation_retrieval(question)
            print('Answer >>>', answer)
            continue
        # There are 2 keywords
        db = model.knowledgeBased_Retrieval_Location(keyword[0], keyword[1])
        # When db return -2 , it does not require to documentation retrieval because the nearest exit have been found
        if(db != -2):
            answer = model.documentation_retrieval(question)
            print('Answer >>>', answer)
            continue


    # Case for asking the route of how to go to one station from another station
    elif (res[0] == 'description') & (res[1] == 'route'):
        start_index = 0
        dest_index = 0    
        if(len(keyword) != 2):
            print('Please give the destination / starting point in the question.')
            continue

        keyword = model.checkStationName([keyword[0], keyword[1]])

        # Finding out the index of starting point and destination
        for key, value in model.station_name.items():
            if value == keyword[0]:
                start_index = key
            if value == keyword[1]:
                dest_index = key
        

        # If there is only one station exist and there is an another destination point exist
        if (start_index == 0) | (dest_index == 0):
            db = model.knowledgeBased_Retrieval_Location(keyword[0], keyword[1])
            if(db != -2):
                answer = model.documentation_retrieval(question)
                print('Answer >>>', answer)
                continue
            continue
        
        # API part
        resp = requests.get('http://www.mtr.com.hk/share/customer/jp/api/HRRoutes/?lang=E', params={'o': start_index, 'd' : dest_index})
        resp = resp.json()
        route = resp['routes'][0]['path']
        walk_interchange = 0
        print('Starting point:', string.capwords(model.station_name.get(start_index)))
        for i in route:
            if (i['linkType'] == 'INTERCHANGE'):
                if walk_interchange == 0:
                    print('Travel to', string.capwords(model.station_name.get(i['ID'])))
                walk_interchange = 0
            if (i['linkType'] == 'WALKINTERCHANGE'):
                print('Travel to', string.capwords(model.station_name.get(i['ID'])))
                walk_interchange = 1
            if i['linkText'] != None:
                print(i['linkText'])
            if i['linkType'] == 'END':
                print('Destination station:', string.capwords(model.station_name.get(i['ID'])))
        print('Estimatede time:', resp['routes'][0]['time'],'mins')


    # Case for documentation retrival
    else:
        answer = model.documentation_retrieval(question)
        print('Answer >>>', answer)