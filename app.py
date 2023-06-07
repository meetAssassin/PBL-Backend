from flask import Flask, request, render_template
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

le = LabelEncoder()
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///review.db'  # SQLite database file
# db = SQLAlchemy(a
@app.route('/')
def home():
    # messages = Message.query.all()
    # return render_template("home.html", messages=messages)
    return render_template("home.html")

@app.route("/predictandTranslate", methods=["POST"])
def predict():
    data = pd.read_csv("language_detection.csv")
    y = data["Language"]
    y = le.fit_transform(y)

    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # preprocessing the text
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        # creating the vector
        vect = cv.transform(dat).toarray()
        # prediction
        my_pred = model.predict(vect)
        my_pred = le.inverse_transform(my_pred)

        # storing the message in the database
        # content = text
        # message = Message(content=content)
        # db.session.add(message)   
        # db.session.commit()

    # return render_template("home.html", pred="The entered text is in {}".format(my_pred[0]))

# @app.route("/translate", methods=["POST"])
# def translate():
    translator = Translator()
    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # taking the target language
        target_lang = request.form["lang"]
        # translating the text
        translation = translator.translate(text, dest=target_lang)
        # return render_template("home.html", translation=translation.text)

        return render_template("home.html", pred="{}".format(my_pred[0]), translation=translation.text)

if __name__ == "__main__":
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)