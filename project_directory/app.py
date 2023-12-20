from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random as rd
app = Flask(__name__)

# Sample Product Data (you would use your own dataset)
products = {
    1: "Laptop",
    2: "Smartphone",
    3: "Headphones",
    4: "Camera",
    5: "Airdopes",
    6: "Clothes",
    7: "Backpack",
}

user_history = {
    "user1": rd.choice(products),
    "user2": rd.choice(products)
}

product_features = {
    1: np.array([1, 0, 1, 0]),
    2: np.array([0, 1, 0, 1]),
    3: np.array([1, 0, 0, 1]),
    4: np.array([0, 1, 1, 0]),
    6: np.array([1, 1, 0, 0]),  # Add the missing product ID
}



# Generate Cosine Similarity Matrix
cosine_sim = cosine_similarity(list(product_features.values()))

def recommend_products(user_history):
    user_vector = np.sum([product_features.get(product, np.zeros(4)) for product in user_history], axis=0)

    # Ensure that user_vector has the same length as the number of columns in cosine_sim
    user_vector = np.append(user_vector, 0)  # Add a placeholder for the missing dimension

    # Debugging: Print dimensions and values
    print("cosine_sim shape:", cosine_sim.shape)
    print("user_vector shape:", user_vector.shape)
    print("user_vector:", user_vector)
    print("cosine_sim:", cosine_sim)

    scores = np.dot(cosine_sim, user_vector)

    # Adjust the product indices to start from 1
    recommended_products = [products.get(i + 1, f"Product {i+1}") for i in np.argsort(scores)[::-1] if i + 1 not in user_history]

    return recommended_products[:3]





# Flask Routes
@app.route('/')
def index():
    return render_template('index.html', products=products)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.form.get('user_id')
    if user_id in user_history:
        recommendations = recommend_products(user_history[user_id])
        return render_template('recommendations.html', recommendations=recommendations)
    else:
        return "User not found"

if __name__ == '__main__':
    app.run(debug=True)
