from typing import List, Tuple

import pandas as pd
import torch
import torch.optim as optim

from dataset import NYTCluesDataset, LETTERS
from model import MutantLetterBinaryClassifier

MAX_EPOCHS = 50
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 1000
LEARNING_RATE = 1e-4
DATALOADER_WORKERS = 4


def initialize_dataset() -> Tuple:
    print("========== Initializing Dataset ==========")
    nyt_clue_data = pd.read_csv(
        'data/nytcrosswords_cleaned.csv',
        usecols=('Word', 'Clue'),
        dtype= {'Word': str, 'Clue': str},
        encoding='utf-8'
    )
    xwi_wordlist_data = pd.read_csv('data/XwiWordList.txt', names=('answer', 'score'), sep=';')

    dataset = NYTCluesDataset(data=nyt_clue_data.head(200000).copy(), scores=xwi_wordlist_data)

    return torch.utils.data.random_split(dataset, [.8, .1, .1]), dataset


def train_model(train: torch.utils.data.DataLoader, test: torch.utils.data.DataLoader) -> MutantLetterBinaryClassifier:
    print("============= Training Model =============")
    xw_model = MutantLetterBinaryClassifier(hidden_layer_size=HIDDEN_LAYER_SIZE).cuda()

    optimizer = optim.Adam(xw_model.parameters(), lr = LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    prev_loss = float('inf')

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch + 1}...")

        for batch, data in enumerate(train):
            optimizer.zero_grad()

            x = torch.stack( [ xw_model.preprocess_input(d) for d in data ] )
            labels = torch.stack( [ torch.tensor( [d['label']] ) for d in data ] ).cuda()

            y = xw_model(x)

            loss = criterion(y, labels)

            loss.backward()
            optimizer.step()

            # if batch % 50 == 0:
            #     print(f"Loss for batch {batch} is {loss.item():.4f}")

        loss = test_model(xw_model, test)
        print(f"Loss for epoch {epoch + 1}: {loss:.4f}")

        if (prev_loss - loss < 0.002):
            break

        prev_loss = loss

    return xw_model


def test_model(xw_model: MutantLetterBinaryClassifier, test: torch.utils.data.DataLoader) -> float:

    criterion = torch.nn.BCELoss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for data in test:
            x = torch.stack( [ xw_model.preprocess_input(d) for d in data ] )

            labels = torch.stack( [ torch.tensor( [d['label']] ) for d in data ] ).cuda()
            y = xw_model(x)

            total_loss += criterion(y, labels).item()

            total_batches += 1

    return total_loss/total_batches


def score_letter(xw_model: MutantLetterBinaryClassifier, dataset: NYTCluesDataset, clue: str, answer: str, index: int) -> float:
    all_mutants = [ answer[:index] + letter + answer[index + 1:] for letter in LETTERS ]
    all_items = [ dataset.make_item(clue, mutant_answer, index, 0) for mutant_answer in all_mutants ]
    min_mutant_score = torch.min( torch.stack( [ xw_model(xw_model.preprocess_input(d)) for d in all_items ] ) ).item()

    orig_score = xw_model(xw_model.preprocess_input(dataset.make_item(clue, answer, index, 0))).item()

    print(f"Min mutant score is {min_mutant_score}")
    print(f"Orig answer score is {orig_score}")

    print(f"Ambiguity is {orig_score - min_mutant_score}")


if __name__ == "__main__":

    model = None

    while (True):
        command = input(" $ ")

        if command == "data":
            (train, val, test), dataset = initialize_dataset()
            
            train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
            val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
            test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

        elif command == "train":
            model = train_model(train_loader, test_loader)
        
        elif command == "examples":
            example_test_batch = next(iter(test_loader))

            for test_input in example_test_batch:
                print(f"Clue: {test_input['clue']}, Answer: {test_input['answer']}")

                x = model.preprocess_input(test_input)
                pred = model(x)
                print(f"-> Score: {pred.item():.4f}, Actual: {test_input['label']}")

        elif command == "test":
            if model == None:
                print("Model not trained. Skipping command.")

            print(f"Test loss: {test_model(model, test_loader):.4f}")

        elif command == "score":
            while True:
                if model == None:
                    print("Model not trained. Skipping command.")

                clue = input(" -> Clue: ")
                answer = input(" -> Answer: ")
                index = int(input(" -> Index: "))

                score_letter(model, dataset, clue, answer, index)

        elif command == "save":
            torch.save(model.state_dict(), 'xw_model.pth')

        elif command == "load":
            model = MutantLetterBinaryClassifier(hidden_layer_size=HIDDEN_LAYER_SIZE)
            model.load_state_dict(torch.load('xw_model.pth'))
            model.cuda()

        elif command == "exit":
            break

        else:
             print(f"Invalid command {command}. Skipping commmand.")
