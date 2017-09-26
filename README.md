# Implementation of Dynamic Memory Network Plus

This is a rough 'untested' implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The answer module is described in this [paper](https://arxiv.org/pdf/1506.07285.pdf).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I don't have enough computational power for training heavy models like this under a reasonable time.

I trained the model for a few iterations (one story per iteration) on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). Some of the iterations with predicted outputs and target outputs are visible in
the ipynb file - it shows the model in some action. However, training is far from complete.

This implementation wasn't tested on validation or test data set.

Caution: I haven't implemented any regularization.






