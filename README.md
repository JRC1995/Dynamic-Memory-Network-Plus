# Implementation of Dynamic Memory Network Plus

This is a rough 'untested' implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The answer module is described in this [paper](https://arxiv.org/pdf/1506.07285.pdf).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I trained the model for a few epochs on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

This implementation wasn't tested on test data set, since I didn't complete the training.

Hyperparameters may vary from the original implementation. 
