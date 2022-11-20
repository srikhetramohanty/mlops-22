from flask import Flask,request,jsonify
import glob
from joblib import load
#from jsonify import jsonify

app = Flask(__name__)

@app.route("/predict",methods=['POST'])
def predict():
    content = request.json
    img1 = content['image']
    model = content['model_name']
    
    if model=="svm":
        best_model = load("./svm_gamma=0.001_C=0.5.joblib")
    elif model=="tree":
        best_model = load("./tree_max_depth_8_Criterion_entropy")

    predicted_digit = best_model.predict([img1])
    
    return jsonify({"predicted_digit":str(predicted_digit[0]),
                    "model":model})

if __name__ == "_main_":
    app.run()