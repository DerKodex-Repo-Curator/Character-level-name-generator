from torch.utils.data import DataLoader, TensorDataset
import torch
import re

with open("names.txt", "r") as f:
  names = f.read()
names = names[1:]

def clean_text(text):
  text = text.lower()
  text = text.replace("* ", "")
  text = text.split("\n")
  for i,names in enumerate(text):
    text[i] = re.sub(r'\s+\w+’*\w+$|\s+', "", names)

  for i,names in enumerate(text):
    text[i] = f"%{names}$"

  return text

names_list = clean_text(names)
names_list += ["!"]
characters = sorted(list(set("".join(names_list))))
char2idx = {c:i for i, c in enumerate(characters)}
idx2char = {i:c for i, c in enumerate(characters)}

encoded = []
for names in names_list[:len(names_list) - 1]:
  encoded.append([char2idx[c] for c in names])

def make_batches(encoded_list):
  input_batch = []
  output_batch = []
  for names in encoded_list:
    input = names[:len(names) - 1] + [0] * (13 - len(names))
    output = names[1:] + [0] * (13 - len(names))
    input_batch.append(input)
    output_batch.append(output)

  input_batch = torch.tensor(input_batch, dtype=torch.long)
  output_batch = torch.tensor(output_batch, dtype=torch.long)

  dataset = TensorDataset(input_batch, output_batch)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

  print(f"Dataloader length {len(dataloader)}")

  return dataloader

dataloader = make_batches(encoded)
