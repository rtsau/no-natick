"""
Inputs to tool:
    clue 1, answer 1, index
    clue 2, answer 2, index

Tool checks whether there is a match at that index, and if there is, feeds the individual (clue, answer, index) tuples to the preprocesser

Inputs to tuple preprocessor:
    (clue, answer, index)

    * tokenize clue
    * tokenize answer

Features to concatenate:

"""

from typing import Dict, Tuple

from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd


from dataset import ANSWER_MAX_LENGTH, GLOVE_EMBEDDING_LENGTH, NGRAM_FREQS_LENGTH


LINEAR1_INPUT_DIMS = ANSWER_MAX_LENGTH + 2*GLOVE_EMBEDDING_LENGTH + NGRAM_FREQS_LENGTH

class MutantLetterBinaryClassifier(nn.Module):
    """Classifier scoring whether a specified tuple represents a mutated crossword answer."""

    def __init__(self, hidden_layer_size: int):
        super().__init__()

        self.linear1 = nn.Linear(LINEAR1_INPUT_DIMS, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, 1)

        self.activation_function = torch.relu

        self.unk_embedding = nn.Embedding(num_embeddings=1, embedding_dim=GLOVE_EMBEDDING_LENGTH)
        self.unk_embedding.weight.data

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear1(input)
        x = self.activation_function(x)
        x = self.linear2(x)
        x = self.activation_function(x)
        x = self.linear3(x)
        y = torch.sigmoid(x)

        return y
    
    def preprocess_input(self, model_input: Dict) -> torch.Tensor:
        ngram_probs = model_input['ngram_probs'].cuda()
        clue_embedding = model_input['glove_clue']['glove_avg'].cuda() + self.unk_embedding.weight.data * model_input['glove_clue']['unk_ratio'].cuda()
        answer_embedding = model_input['glove_answer']['glove_avg'].cuda() + self.unk_embedding.weight.data * model_input['glove_answer']['unk_ratio'].cuda()
        index_one_hot = model_input['index_one_hot'].cuda()

        return torch.cat( [ngram_probs, clue_embedding.squeeze(), answer_embedding.squeeze(), index_one_hot] )
        