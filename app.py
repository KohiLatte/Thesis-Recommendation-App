from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import redirect
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote


app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

os.chdir(r'C:\Users\arita\Desktop\Rec App') # set the working directory
df = pd.read_json('intents.json') # read the JSON file containing data

def find_similar(vector_representation, all_representations, k=1):
    # function to find the most similar paragraph to the search query
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])

paragraph = df.iloc[:, 0] # the first column values
embeddings_distilbert = model.encode(paragraph.values) # encode the paragraphs using the pre-trained SentenceTransformer model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_string = request.form["search_string"] # get the search query from the form on the index.html page

        if not search_string:
            return "query parameter is required"

        search_vect = model.encode([search_string]) # encode the search query using the pre-trained SentenceTransformer model
        output_data = [paragraph[i] for i in find_similar(search_vect, embeddings_distilbert)][0] # find the most similar paragraph to the search query

        tag = output_data["tag"]
        responses = output_data["responses"]

        # Create Lazada search link
        lazada_search_url = f"https://www.lazada.com.ph/catalog/?q={quote(tag)}" # create a Lazada search link using the tag associated with the most similar paragraph

        return render_template("index.html", result=responses, lazada_search_url=lazada_search_url) # render the index.html page with the responses and Lazada search link
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) # run the Flask app in debug mode
