# Xword-Difficulty: Crossword Difficulty Evaluation via Anomaly Classification

This project trains and runs a model that provides difficulty scores for crosswords. Specifically, the model can provide scores for a specific index in a crossword answer at a particular index given some clue.

## Quick-start guide

To run this project, install the required packages and run `python3 run.py`. Once running, you will see a prompt `$` and can give one of several commands:

* `data`: loads data into the model datasets from the data/ directory. Must be done before running any other commands, because they all rely on some information harvested during the data preprocessing.
* `train`: trains the model to completion and provides some output describing this process.
* `examples`: prints out some example inputs and their output scores.
* `test`: runs the model on test set and provides loss information.
* `score`: provides an interface that will provide scores for user-inputted inputs (will ask for a clue, answer (should be all uppercase), and index within the answer).

## Acknowledgements

This project includes code from the following open-source libraries:
* [PuzPy](https://github.com/alexdej/puzpy) by Alex Dejarnatt