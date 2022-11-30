from flask import Flask,render_template,request

import pickle as pkl

from flask import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)




#load model
filename='tfvectorizer_features.pkl'
tf_idf_converter = pkl.load(open(filename, 'rb'))

filename='xgb(IE).pkl'
model1 = pkl.load(open(filename, 'rb'))

filename='xgb(NS).pkl'
model2 = pkl.load(open(filename, 'rb'))

filename='xgb(FT).pkl'
model3 = pkl.load(open(filename, 'rb'))

filename='xgb(JP).pkl'
model4 = pkl.load(open(filename, 'rb'))
@cross_origin(supports_credentials=True)

@app.route('/predict',methods=['POST','GET'])
def predict():
    b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
    my_posts=["""hi im am a student of computer science i wan to persue master in related field and excel in it so i can give something to the society"""]
    result = []
    prediction=model1.predict(tf_idf_converter.transform(my_posts))
    print(prediction)
    result.append(prediction[0])

    prediction=model2.predict(tf_idf_converter.transform(my_posts))
    print(prediction)
    result.append(prediction[0])

    prediction=model3.predict(tf_idf_converter.transform(my_posts))
    print(prediction)
    result.append(prediction[0])

    prediction=model4.predict(tf_idf_converter.transform(my_posts))
    print(prediction)
    result.append(prediction[0])

    print(result)
        # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(result):
        s += b_Pers_list[i][l]
    print("The result is: ", s)
    return s





if __name__=='__main__':
    app.run(debug=True)