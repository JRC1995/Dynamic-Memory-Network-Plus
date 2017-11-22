# Implementation of Dynamic Memory Network Plus

This is a rough 'untested' implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The answer module is described in this [paper](https://arxiv.org/pdf/1506.07285.pdf).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I trained the model for a few epochs on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

Hyperparameters may vary from the original implementation.


Hidden size used = 100

Embedding size used = 100


I am training the model to be weakly supervised. That is the model won't be told which supporting facts are relevant for inductive reasoning in order to derive an answer. 

The published classification error of QA task 16 (basic induction) of bAbi Dataset of the DMN+ model (as given here: https://arxiv.org/pdf/1603.01417.pdf) is 45.3% 

My implementation of the model on pretrained 100 dimensional GloVe vectors seems to produce about 49.1% classification accuracy on Test Data for induction tasks...i.e the classification error is about 50.9%. The error less than what the original DMN model acheived as specified in the paper, but still greater than the errors achieved achieved by the improved versions of DMN (DMN1, DMN2, DMN3).






