from flask import Flask, jsonify, request, render_template, json
import backend_server
import requests
import string
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/questions/<string:question>', methods =['GET'])
def getQuestion(question):
    # Check whether it is a valid question
    if(len(question.split(' ')) <= 1):
        return model.generate_respond(-1, ['Not a question, please reinput again !'], [''])  

    # Predict the type of the question
    res = model.transfer_type(question)
    print('Type >>>', res)

    # keyword searching (for all cases except documentation retrieval method)
    keyword = model.questionHandler(question)

    # Case for getting price of the ticket
    if(res[1] == 'money'):
        return model.priceHandler(keyword)

    # Case for asking toilet in MTR station (frequent term of describe 'toilet')
    elif (('wc' in [item.lower() for item in question.split(' ')])| ('toilet' in [item.lower() for item in question.split(' ')]) | ('toilets' in [item.lower() for item in question.split(' ')])):
        return model.toiletHandler(keyword)

    # Case for asking the route of how to go to one station from another station
    elif (res[0] == 'description') & (res[1] == 'route') | (res[0] == 'location') & (res[1] == 'other'):
        return model.locationHandler(keyword, question, res[1])

    # Case for documentation retrival
    else:
      answer = model.documentation_retrieval(question, res[1])
      if (answer != []):
        return model.generate_respond(1, answer, [''])
      else:
        return model.generate_respond(-1, ['Question does not related to MTR'], [''])


if __name__ == '__main__':
  global model
  model = backend_server.SVM_Linear()
  app.run(port = 5000)