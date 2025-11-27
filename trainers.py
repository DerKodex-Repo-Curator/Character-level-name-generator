from make_data import dataloader, characters
from tqdm.auto import tqdm
import torch.nn as nn
import torch

def train_RNN(model_class):
    vocab_size = len(characters)
    hidden_size = 128
    embed_size = 100
    learning_rate = 0.005
    epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelRNN = model_class(vocab_size, hidden_size, embed_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(modelRNN.parameters(), lr=learning_rate)

    print("Starting Training")

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for input_batch, output_batch in dataloader:
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            batch_size = input_batch.size(0)
            hidden = modelRNN.init_hidden(batch_size).to(device)

            optimizer.zero_grad()
            output, hidden = modelRNN(input_batch, hidden)

            loss = criterion(output.reshape(-1, vocab_size), output_batch.reshape(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
            
    print("Training Ended")
    return modelRNN
            
def train_LSTM(model_class):
    vocab_size = len(characters)
    hidden_size = 128
    embed_size = 100
    learning_rate = 0.005
    epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelLSTM = model_class(vocab_size, hidden_size, embed_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(modelLSTM.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)

            hidden = modelLSTM.init_hidden(input.size(0))
            hidden = (hidden[0].to(device), hidden[1].to(device))

            optimizer.zero_grad()

            output, hidden = modelLSTM(input, hidden)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(modelLSTM.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")
            
    print("Training Ended")
    return modelLSTM
            
def train_GRU(model_class):
    vocab_size = len(characters)
    hidden_size = 128
    embed_size = 100
    learning_rate = 0.005
    epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelGRU = model_class(vocab_size, hidden_size, embed_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(modelGRU.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)

            hidden = modelGRU.init_hidden(input.size(0)).to(device)

            optimizer.zero_grad()

            output, hidden = modelGRU(input, hidden)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(modelGRU.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")
            
    print("Training Ended")
    return modelGRU