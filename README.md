# Implementation of Dynamic Memory Network Plus

This is a rough 'untested' implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The answer module is described in this [paper](https://arxiv.org/pdf/1506.07285.pdf).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).

I don't have enough computational power for training heavy models like this under a reasonable time.

I trained the model for a few iterations (one story per iteration) on solely induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). Some of the iterations with predicted outputs and target outputs are visible in
the ipynb file - it shows the model in some action. However, training is far from complete.

This implementation wasn't tested on validation or test data set.

Here are some of the training iterations:


>Iter 0, Loss= 12.458, Predicted Answer= vlaeminck, Actual Answer= green <br/>
>Iter 100, Loss= 3.438, Predicted Answer= yellow, Actual Answer= green <br/>
>Iter 200, Loss= 1.613, Predicted Answer= gray, Actual Answer= white <br/>
>Iter 300, Loss= 3.093, Predicted Answer= yellow, Actual Answer= white <br/>
>Iter 400, Loss= 2.933, Predicted Answer= gray, Actual Answer= white <br/>
>Iter 500, Loss= 0.767, Predicted Answer= yellow, Actual Answer= yellow

I jupyter notebook version looks ugly on github. I commented it to death (didn't know about ability to insert markdown cells in Jupyter back then).

The .py version of the same code may appear aesthetically slighly more pleasing. 
