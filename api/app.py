from flask import Flask
from flask import request
from joblib import load
import jsonify

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Mlops-Class-IITJ </b>"



@app.route("/predict", methods=['GET','POST'])
def predict():
    content = request.json
    img1 = content['image1']

    img2 = content['image2']
    
    predicted__1 = model.predict([img1])
    predicted__2 = model.predict([img2])
    if predicted_1 == predicted_2:
        is_same = True
    else:
        is_same = False
    return jsonify({"predicted_1":str(predicted_1[0]),
                    "predicted_2":str(predicted_2[0]),
                    "is_same":is_same})

    if _name_ == "_main_":
        app.run(port=5000)

# def predict_digit():
#     image = request.json['image']
#     print("done loading")
#     predicted = model.predict([image])
#     return {"y_predicted":int(predicted[0])}