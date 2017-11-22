# Implementation of Dynamic Memory Network Plus

This is a rough 'untested' implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The answer module is described in this [paper](https://arxiv.org/pdf/1506.07285.pdf).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I trained the model for a few epochs on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

This implementation wasn't tested on test data set, since I didn't complete the training.

Hyperparameters may vary from the original implementation. I used orthogonal initialization on the hidden state weights. That seemed to speed up training. 

Hidden size used = 100
Embedding size used = 100

I am training the model to be weakly supervised. That is the model won't be told which supporting facts are relevant for inductive reasoning in order to derive an answer. 

The published classification error of QA task 16 (basic induction) of bAbi Dataset of the DMN+ model (as given here: https://arxiv.org/pdf/1603.01417.pdf) is 45.3. 

My model's validation classification error seems to be approaching there. 



