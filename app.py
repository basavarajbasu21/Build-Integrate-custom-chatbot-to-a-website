from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json
import random

app = Flask(__name__)
CORS(app)

# Load the intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model with error handling
FILE = "data.pth"
try:
    data = torch.load(FILE, weights_only=True)
except Exception as e:
    print(f"Warning: Loading the model with weights_only=True failed: {e}")
    print("Attempting to load the model with weights_only=False. Ensure the file is from a trusted source.")
    try:
        data = torch.load(FILE)
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.json['message']
    except KeyError:
        return jsonify({"error": "No message field provided"}), 400

    response = get_response(message)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)