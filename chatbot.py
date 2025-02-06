from flask import Flask, request, jsonify
import torch
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Hi, how are you?",
    "What is your name?",
    "Tell me a joke.",
    "What is the weather today?",
    "How can I help you?",
    "Goodbye!"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)  # 'corpus' is your list of sentences


from chatbot_model import ChatbotModel  # Ensure your model is defined in chatbot_model.py

# Initialize Flask app
app = Flask(__name__)

# Define the model architecture (if not already done in chatbot_model.py)
class ChatbotModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and load trained weights
model = ChatbotModel(input_size=300, hidden_size=128, output_size=6)  # Adjust as per your model
# Check the architecture of the saved model
model_checkpoint = torch.load("chatbot_model.pth")
print(model_checkpoint.keys())

# LabelEncoder for categories
with open("intents.json", "r") as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return words

# Define route for chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Preprocess input
    words = preprocess_input(user_input)

    # Convert words to vectors (you should have a function to convert words to vectors)
    X_input = vectorizer.transform([words]).toarray()  # Assuming you have a fitted vectorizer

    # Convert to torch tensor
    X_input_tensor = torch.tensor(X_input, dtype=torch.float32)

    # Get model prediction
    with torch.no_grad():
        output = model(X_input_tensor)
        _, predicted = torch.max(output, dim=1)
    
    # Get predicted category label
    category = LabelEncoder.inverse_transform([predicted.item()])

    # Find the response based on predicted category
    response = get_response_from_category(category[0])  # You should implement this

    return jsonify({"response": response})

# Function to return chatbot response based on category
def get_response_from_category(category):
    for intent in intents['intents']:
        if intent['tag'] == category:
            return np.random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

if __name__ == "__main__":
    app.run(debug=True)

