from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import faiss

# ----------------------------------------------------
products = pd.read_csv("products_cleaned.csv")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

tfidf_vectors = np.load("tfidf_vectors.npy").astype("float32")
d = tfidf_vectors.shape[1]   
index = faiss.IndexFlatIP(d)
index.add(tfidf_vectors)

# ----------------------------------------------------

app = Flask(__name__)
    

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"].strip().lower()
    
    if not query:
        return render_template("index.html", error="Please enter a search query")
    
    mask = products["clean_name"].str.contains(query, na=False)
    results = products[mask].head(12)
    
    if len(results) == 0:
        
        mask = products["name"].str.contains(query, case=False, na=False)
        results = products[mask].head(12)
    
    return render_template("results.html", items=results, query=query)


@app.route("/recommend/<product_name>")
def recommend(product_name):
    product_idx = products[products["name"].str.lower() == product_name.lower()].index
    
    if len(product_idx) == 0:
        product_idx = products[products["name"].str.contains(product_name, case=False, na=False)].index
    
    if len(product_idx) == 0:
        return "Product not found", 404
    
    pid = product_idx[0]
    query_vec = tfidf_vectors[pid:pid+1]
    scores, ids = index.search(query_vec, 12)
    items = products.iloc[ids[0]]
    
    product_name = products.iloc[pid]["name"]
    
    return render_template("results.html", items=items, query=f"Similar to: {product_name}")


# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)