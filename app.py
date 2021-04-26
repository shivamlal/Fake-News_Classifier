from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


#### Load the model ####

filename = 'model.pkl'
clsfr = pickle.load(open(filename, 'rb'))
tfidf_vc = pickle.load(open('vect.pkl', 'rb'))
app = Flask(__name__)


app = Flask(__name__)


@app.route ('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form["type"]
        data = [news]
        vect = tfidf_vc.transform(data).toarray()
        my_prediction = clsfr.predict(vect)
        return render_template('result.html', prediction = my_prediction)
    else:
        return "Something went worng"    






if __name__ == "__main__":
    app.run(debug = True)
    