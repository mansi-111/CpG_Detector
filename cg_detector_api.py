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
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional layer for embedding

class CPGCounter(nn.Module):
  def __init__(self, input_size, embedding_dim, hidden_size,num_layers,dropout_prob =0.2):
    super(CPGCounter, self).__init__()
    self.embedding = nn.Embedding(input_size, embedding_dim,padding_idx=0)  # Define embedding layer
    # self.lstm = nn.LSTM(embedding_dim, hidden_size)
    self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,dropout=dropout_prob, batch_first=True)
    self.linear = nn.Linear(hidden_size, 1)

  def forward(self, x):
    # Embed DNA characters
    embeddings = self.embedding(x)

    # Pass through LSTM
    lstm_out, _ = self.lstm(embeddings)

    # Get last output and predict count
    output = self.linear(lstm_out[:, -1, :])
    return output
  
model_emb = torch.load("model_with_embedding/CP_count_model_padding_with_emb.pth")

def model_prediction_with_emb(input_string):
    # Save the model
    
    nucleotide_to_index = {'N': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, 'pad': 0}

    # Function to convert a string of nucleotides to a tensor
    def string_to_tensor(s, mapping):
        max_length = 128
        # Convert the string to a list of integers
        indices = [mapping[c] for c in s]
        padding_length = max_length - len(indices)
        indices.extend([0] * padding_length)  # Pad with
        tensor = torch.LongTensor([indices])
        return tensor
    test_tensor = string_to_tensor(input_string, nucleotide_to_index)
    output = model_emb(test_tensor)
    output = output.squeeze().tolist()
    return output

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
    output = model_prediction_with_emb(input_string)
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
