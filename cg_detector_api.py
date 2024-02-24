from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torch.nn as nn

class CpGPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=1, output_size=1):
        super(CpGPredictor, self).__init__()
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define the classifier layer
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states for each batch
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.classifier(out[:, -1, :])
        return out
    
app = Flask(__name__)

def model_prediction(input_string):
    # Define the mapping from nucleotides to integers
    nucleotide_to_index = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

    # Function to convert a string of nucleotides to a tensor
    def string_to_tensor(s, mapping):
        indices = [mapping[c] for c in s]
        tensor = torch.tensor(indices)
        one_hot = F.one_hot(tensor, num_classes=len(mapping))
        one_hot = one_hot.unsqueeze(0).float()
        return one_hot
    # Convert the test string to a tensor
    test_tensor = string_to_tensor(input_string, nucleotide_to_index)

    # Load the model (assuming it's the same as before)
    model = CpGPredictor()
    state_dict = torch.load("CP_count_model.pth")

    # Remove the unexpected keys from the state dictionary
    unexpected_keys = ['lstm.weight_ih_l1', 'lstm.weight_hh_l1', 'lstm.bias_ih_l1', 'lstm.bias_hh_l1']
    for key in unexpected_keys:
        if key in state_dict:
            print("key--------",key)
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Pass the tensor through the model
    with torch.no_grad():
        prediction = model(test_tensor)

    # Print the prediction
    print(prediction)
    prediction = prediction.item()
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input string from the form
    input_string = request.form['input_string']
    output = model_prediction(input_string)
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)