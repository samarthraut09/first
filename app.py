from flask import Flask,render_template,request
import pickle
import numpy as np

Logistic_model = pickle.load(open('Logistic_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    # prediction
    result = Logistic_model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'Not placed'

    return render_template('home.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)