import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Dummy dataset
texts = [
    "Ich liebe es, in den Urlaub zu fahren!",
    "Das Wetter ist heute schrecklich.",
    "Ich fühle mich neutral über das Thema."
]

labels = [1, 0, 1]  # 1 = positiv, 0 = negativ

# Trainiere das Modell
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

def analysiere_sentiment(text):
    prediction = model.predict([text])
    return "Positiv" if prediction[0] == 1 else "Negativ"

if __name__ == '__main__':
    text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Bitte geben Sie einen Text ein: ')
    sentiment = analysiere_sentiment(text)
    print(f'Das Sentiment des eingegebenen Textes ist: {sentiment}')