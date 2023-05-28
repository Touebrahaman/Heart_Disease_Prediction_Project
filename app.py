from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    
    #output=round(prediction[0], 2)

    if prediction==1:
        return render_template('index.html',pred='More chances of Heart Attack so the result is --->> {}'.format(prediction))
    else:
        return render_template('index.html',pred='Less chances of Heart Attack so the result is --->> {}'.format(prediction))
    


if __name__ == '__main__':
    app.run(debug=True)