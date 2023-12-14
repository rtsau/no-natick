import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import gensim.downloader
from typing import Dict, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from nltk.tokenize import word_tokenize

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NGRAM_SAMPLE_SIZE = 10000
ANSWER_MAX_LENGTH = 22
GLOVE_EMBEDDING_LENGTH = 100
NGRAM_FREQS_LENGTH = 6

class NYTCluesDataset(Dataset):
    """Dataset containing clues and answers, alongside clues and fake
    answers (created by randomly changing one character in the real answer)."""


    def __init__(self, data: pd.DataFrame, scores: pd.DataFrame):
        super().__init__()

        data['Clue'] = data['Clue'].astype(str)
        data['Word'] = data['Word'].astype(str)

        self.calculate_ngram_freqs(data=data)

        self.glove = gensim.downloader.load('glove-wiki-gigaword-100') # TODO: try other dimensions
        # print(f"ashed glove is {self.glove['ashed']}")

        random.seed(0)
        self.data = []

        for i, (clue, answer) in enumerate(zip(data['Clue'], data['Word'])):
            index = random.randint(0, len(answer) - 1)

            # TODO: sample previously observed trigrams to get new letter, not just random
            
            label = 0.0
            if i % 2 == 0:
                label = 1.0
                answer = answer[:index] + random.choice(LETTERS) + answer[index + 1:]

            item = self.make_item(clue, answer, index, label)

            self.data.append(item)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]
    
    def calculate_ngram_freqs(self, data) -> None:
        """Calculates frequencies of bigrams and trigrams."""

        sampled = '^' + data.sample(n=NGRAM_SAMPLE_SIZE, random_state=0) + '^'

        bigram_vec = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        trigram_vec = CountVectorizer(analyzer='char', ngram_range=(3, 3))

        bigram_freqs = pd.DataFrame(bigram_vec.fit_transform(sampled['Word']).toarray(), columns=bigram_vec.get_feature_names_out())
        trigram_freqs = pd.DataFrame(trigram_vec.fit_transform(sampled['Word']).toarray(), columns=trigram_vec.get_feature_names_out())

        self.bigram_freqs = defaultdict(int, bigram_freqs.sum(axis=0).to_dict())
        self.trigram_freqs = defaultdict(int, trigram_freqs.sum(axis=0).to_dict())

        self.bigram_pat_freqs = defaultdict(int)
        self.trigram_pat_freqs = defaultdict(int)
        
        for bigram, freq in self.bigram_freqs.items():
            self.bigram_pat_freqs[bigram[0] + '*'] += freq
            self.bigram_pat_freqs['*' + bigram[1]] += freq
        
        for trigram, freq in self.trigram_freqs.items():
            self.trigram_pat_freqs[trigram[0] + '*' + trigram[2]] += freq

    def get_ngram_probs(self, answer: str, index: int) -> torch.Tensor:
        """Returns a tensor representing the likelihood of the character at the given index
        appearing there based on pre-calculated frequencies."""

        answer = '^' + answer.lower() + '^' # pad with start/end chars
        index += 1 # account for padding we just added

        preceded_by = [ self.bigram_pat_freqs[ answer[index - 1] + '*' ], self.bigram_freqs[ answer[index - 1 : index + 1] ] ]
        followed_by = [ self.bigram_pat_freqs[ '*' + answer[index + 1] ], self.bigram_freqs[ answer[index : index + 2] ] ]
        surrounded_by = [ self.trigram_pat_freqs[ answer[index - 1] + '*' + answer[index + 1] ], self.bigram_freqs[ answer[index - 1 : index + 2] ] ]

        return torch.tensor( preceded_by + followed_by + surrounded_by ) / NGRAM_SAMPLE_SIZE
    
    def get_glove_embeddings(self, string: str) -> Dict:
        """Returns a dict containing the averaged tensor representation of the words in `string`
        after tokenization and the proportion of words that correspond to an `UNK` token for
        use with the `UNK` nn.Embedding defined in the model."""

        # print(string)
        tokenized = [token.lower() for token in word_tokenize(string) if token.isalpha()]

        if len(tokenized) == 0:
            return {
                'glove_avg': torch.zeros(GLOVE_EMBEDDING_LENGTH),
                'unk_ratio': torch.tensor( [1] ),
            }
        
        # print(tokenized)
        valid_embeddings = [ torch.tensor(self.glove[token]) for token in tokenized if token in self.glove ]
        
        if len(valid_embeddings) != 0:
            valid_avg = torch.mean(torch.stack(valid_embeddings, dim=0), dim=0)
        else:
            valid_avg = torch.zeros(GLOVE_EMBEDDING_LENGTH)

        unk_ratio = torch.tensor( [1 - len(valid_embeddings)/len(tokenized)] )

        return {
            'glove_avg': (1 - unk_ratio) * valid_avg,
            'unk_ratio': unk_ratio,
        }
    
    def get_one_hot(self, index: int) -> torch.Tensor:
        one_hot = torch.zeros(ANSWER_MAX_LENGTH)
        one_hot[index] = 1

        return one_hot
    
    def make_item(self, clue: str, answer: str, index: int, label: float) -> Dict:
        ngram_probs = self.get_ngram_probs(answer=answer, index=index)
        glove_clue = self.get_glove_embeddings(string=clue)
        glove_answer = self.get_glove_embeddings(string=answer)
        index_one_hot = self.get_one_hot(index=index)

        item = {
            'clue': clue,
            'answer': answer,
            'ngram_probs': ngram_probs,
            'glove_clue': glove_clue,
            'glove_answer': glove_answer,
            'index_one_hot': index_one_hot,
            'label': label,
        }

        return item
