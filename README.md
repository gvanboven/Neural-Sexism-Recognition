## Automated Sexism Recognition

This repository contrains my code and report for the task of automated sexism recognition.  
I carry out this task by:   
i) implementing a neural network from scratch, based on the book "Make  your  own  neural  network book" by Rashid (2016).  
ii) training this neural network on the Sexist Workplace Statements Dataset (Grosz & Conde-Cespedes, 2020).   
iii) I evaluate the performance in my report.   
This project was done for the course 'Machine Learning for NLP' by Antske Fokkens and José Angel Daza at the Vrije Universiteit Amsterdam. 

The file `NN.py` contains the script I adapted from the "Make  your  own  neural  network book" to build a Neural Network. 

I used this script in the file `Sexism_Experiments.ipynb`, which contains my experiments for the task of automated sexism recognition.   
In this file I use pretrained W2V and GloVe embeddings, which should be downloaded (from https://code.google.com/archive/p/word2vec/) and placed in the `./models` folder.