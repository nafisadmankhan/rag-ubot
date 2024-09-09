from flask import Flask, render_template, request
import json

import pandas as pd
import torch
import os
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>We are using {device} device.<<")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

df_filtered = pd.read_csv('/workspaces/codespaces-flask/news_articles_filtered.csv')

with open('/workspaces/codespaces-flask/article_embeddings.pkl', 'rb') as f:
    article_embeddings = pickle.load(f)

dimension = len(article_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(article_embeddings))

def nahidOrg_retrieve_documents(query_embedding, k):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

def nahidOrg_generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    articles = df_filtered['Article text'].tolist()
    article_embeddings = []
    for doc in tqdm(articles, desc="Generating Embeddings For All Articles:"):
        article_embeddings.append(nahidOrg_generate_embeddings(doc))
question = "What is the estimated financial impact on products due to the ban on Russian steel imports, according to the Commission?"
question_embeddings = nahidOrg_generate_embeddings(question)

app = Flask(__name__)

# Initialize the counter variable
counter = 0  

@app.route("/")
def hello_world():
    return render_template("index.html", title="Hello")

@app.route("/uBot")
def uBot():
    return render_template("uBot.html", title="Hello")

@app.route("/run_python", methods=["GET", "POST"])  # Allow both GET and POST requests
def run_python():
    global counter
    counter += 1

    if request.method == "POST":
        # Access incoming data if needed
        data = request.get_json()  # Get the JSON data sent from the client
        print(data['exampleKey'])  # For debugging; you can remove this in production
    
    response = {"count": counter, "exampleValue": "Hi From Python"}
    return json.dumps(response)

