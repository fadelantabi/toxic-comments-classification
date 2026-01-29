# C++ Toxic Comment Classifier

A lightweight Neural Network written in pure C++ (STL) to detect toxic comments. It uses TF-IDF vectorization and a custom Feed-Forward Neural Network built from scratch.

## How to Run

1. **Download Data**: 
   Download `train.csv`,`test.csv` and `test_labels.csv` from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in this folder.

2. **Compile**:
   ```bash
   g++ -O3 main.cpp -o classifier