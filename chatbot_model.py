import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple FeedForward Neural Network model
class ChatbotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChatbotModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the model weights from the saved file
def load_model(model, model_path="chatbot_model.pth"):
    model.load_state_dict(torch.load(model_path))
    model.eval()

# Save the model's state_dict
def save_model(model, model_path="chatbot_model.pth"):
    torch.save(model.state_dict(), model_path)
