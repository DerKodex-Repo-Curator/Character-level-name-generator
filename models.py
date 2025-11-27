import torch.nn as nn
import torch

class NameRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size, embed_size):
    super(NameRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x, hidden):
    x = self.embedding(x)
    output, hidden = self.rnn(x, hidden)
    output = self.fc(output)
    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)

class NameLSTM(nn.Module):
  def __init__(self, vocab_size, hidden_size, embed_size):
    super(NameLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x, hidden):
    x = self.embedding(x)
    output, hidden = self.lstm(x, hidden)
    output = self.fc(output)
    return output, hidden

  def init_hidden(self, batch_size):
    h = torch.zeros(1, batch_size, self.hidden_size)
    c = torch.zeros(1, batch_size, self.hidden_size)
    return (h, c)

class NameGRU(nn.Module):
  def __init__(self, vocab_size, hidden_size, embed_size):
    super(NameGRU, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x, hidden):
    x = self.embedding(x)
    output, hidden = self.gru(x, hidden)
    output = self.fc(output)
    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)