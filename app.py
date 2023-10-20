from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        text = request.form["text"].strip()
        if text:
             sentiment_scores = sid.polarity_scores(text)
        # Determine sentiment based on compound score
             if sentiment_scores["compound"] >= 0.05:
                  sentiment = "Positive"
             elif sentiment_scores["compound"] <= -0.05:
                  sentiment = "Negative"
             else:
                sentiment = "Neutral"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
