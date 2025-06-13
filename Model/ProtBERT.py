import sys

import pandas as pd
from transformers import BertModel, BertTokenizer
import re, os
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert/")
model = BertModel.from_pretrained("Rostlab/prot_bert")


def protbert(file):
    fasta = readFasta(file)
    sequences = []
    for i in fasta:
        name, sequence = i[0], re.sub('-', '', i[1])
        sequences.append(sequence.replace('', ' '))
    encoded_sequences = []
    for sequence in sequences:
        inputs = tokenizer(sequence, max_length=512, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        encoded_sequence = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        encoded_sequences.append(encoded_sequence)
    return np.array(encoded_sequences)

def save_to_csv(encoded_sequences, output_file):
   
    # df = pd.DataFrame(encoded_sequences, columns=['Name', 'Encoding'])
    df = pd.DataFrame(encoded_sequences)
   
    # df['Encoding'] = df['Encoding'].apply(lambda x: x.tolist())  
    df.to_csv(output_file, index=False) 

if __name__ == "__main__":
    input_file = "../Data/data1.txt"  
    output_file = "../Data/data1.csv" 
    encoded_sequences = protbert(input_file)
    save_to_csv(encoded_sequences, output_file)

print(encoded_sequences.shape)  # 如果是 numpy 数组

