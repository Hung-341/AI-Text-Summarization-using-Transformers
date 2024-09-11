# src/ui/app.py

from flask import Flask, request, jsonify, render_template
from src.models.model import TextSummarizer

app = Flask(__name__)

# Load the pre-trained summarization model
summarizer = TextSummarizer(model_name='facebook/bart-large')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get text from the form
        text = request.form['text']

        # Summarize the input text
        summary = summarizer.summarize(text)

        # Return the summarized text in the same page
        return render_template('index.html', text=text, summary=summary)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
