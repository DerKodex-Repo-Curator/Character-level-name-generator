from make_data import char2idx, idx2char
import torch.nn.functional as F
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_name(model, model_type, start_char='%', end_char='$', max_len=20):
    model.eval()
    current_char = start_char
    name = []

    if model_type.upper() in ["RNN", "GRU"]:
      hidden = model.init_hidden(1)
    elif model_type.upper() == "LSTM":
      hidden = model.init_hidden(1)
      hidden = (hidden[0].to(device), hidden[1].to(device))

    with torch.no_grad():
        for _ in range(max_len):
            idx = torch.tensor([[char2idx[current_char]]]).to(device)
            output, hidden = model(idx, hidden)
            probs = F.softmax(output[0, 0], dim=0)
            char_idx = torch.multinomial(probs, 1).item()
            current_char = idx2char[char_idx]
            if current_char == end_char:
                break
            name.append(current_char)
    return "".join(name)

def generate_n_names(model, model_type, n = 15):
    names = []
    for _ in range(15):
        name = generate_name(model, model_type)
        names.append(name)
        n -= 1
    return names

