from models import NameRNN, NameGRU, NameLSTM
from trainers import train_RNN, train_LSTM, train_GRU
from utils import generate_n_names, char2idx
import pandas as pd
import torch

train = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = len(char2idx)
hidden_size = 128
embed_size = 100

modelRNN = NameRNN(vocab_size, hidden_size, embed_size).to(device)
modelLSTM = NameLSTM(vocab_size, hidden_size, embed_size).to(device)
modelGRU = NameGRU(vocab_size, hidden_size, embed_size).to(device)

if train:
  modelRNN = train_RNN()
  modelLSTM = train_LSTM()
  modelGRU = train_GRU()
else:
  modelRNN.load_state_dict(torch.load("RNN_model.pth"))
  modelLSTM.load_state_dict(torch.load("LSTM_model.pth"))
  modelGRU.load_state_dict(torch.load("GRU_model.pth"))

generated_names = pd.DataFrame()
generated_names["RNN names"] = generate_n_names(modelRNN, "RNN")
generated_names["LSTM names"] = generate_n_names(modelLSTM, "LSTM")
generated_names["GRU names"] = generate_n_names(modelGRU, "GRU")

print("Comparison between all model's generated names")
print(generated_names)

generated_names.to_csv("generated_names.csv", index = False)