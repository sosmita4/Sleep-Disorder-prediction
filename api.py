from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np

def return_prediction(model, scaler, data):
    c= list(data.values())
    e= np.array(c,dtype=int)
    w= scaler.transform([e])
    res= model.predict(w)
    label=["Insomnia","None","Sleep Apnea"]
    return label[res[0]]

s=joblib.load('Models/SCALAR.pkl')
rfc_model=joblib.load('Models/RFC.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'

@app.route('/prediction', methods=['POST'])
def predict_disorder():
    # RECIEVE THE REQUEST
    content = request.json

    # PRINT THE DATA PRESENT IN THE REQUEST
    print("[INFO] Request: ", content)

    # PREDICT THE CLASS USING HELPER FUNCTION
    results = return_prediction(model=rfc_model,
                                scaler=s,
                                data =content)

    # PRINT THE RESULT
    print("[INFO] Responce: ", results)

    # SEND THE RESULT AS JSON OBJECT
    return jsonify(results)


if __name__ == '__main__':
    app.run("0.0.0.0")
