# Implementation of Dynamic Memory Network Plus on Tensorflow

This is a rough 'untested' implementation Dynamic Memory Network Plus using tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). 

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I don't have enough computational power for training heavy models like this under a reasonable time.

I trained the model for a few iterations (one story per iteration) on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). The some of the iterations with predicated and target outputs are visible in
the ipynb file. 

So, in summary the model is 'untested' because it was not completely trained, it was not cross validated, and it wasn't
run on test data set. 





