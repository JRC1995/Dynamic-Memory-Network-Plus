# Implementation of Dynamic Memory Network Plus

Implementation of Dynamic Memory Network Plus (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The original Dynamic Memory Network is introduced in this [paper](https://arxiv.org/pdf/1506.07285.pdf) (I had to refer to this paper too).

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).
I used the 100 dimensional embeddings. 

I trained the model on basic induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

Hyperparameters are different from the original implementation.

Hidden size used = 100

Embedding size used = 100


I trained the model in a weakly supervised fashion. That is, the model won't be told which supporting facts are relevant for inductive reasoning in order to derive an answer. 

The published classification error of QA task 16 (basic induction) of bAbi Dataset of the DMN+ model (as given here: https://arxiv.org/pdf/1603.01417.pdf - page 7) is 45.3. 

Why error so high on basic induction?

From the paper:

>One notable deficiency in our model is that of QA16: Basic
Induction. In Sukhbaatar et al. (2015), an untied model
using only summation for memory updates was able to
achieve a near perfect error rate of 0.4. When the memory
update was replaced with a linear layer with ReLU activation,
the end-to-end memory network’s overall mean error
decreased but the error for QA16 rose sharply. Our model
experiences the same difficulties, suggesting that the more
complex memory update component may prevent convergence
on certain simpler tasks.

My implementation of the model on pretrained 100 dimensional GloVe vectors seems to produce about 49.1% classification accuracy on Test Data for induction tasks (check DMN+.ipynb)...i.e the classification error is about 50.9. 


The error is less than what the original DMN model acheived (error 55.1) as specified in the paper, but still greater than the errors achieved achieved by the original implementation of the improved versions of DMN (DMN1, DMN2, DMN3, DMN+) in the paper.


This could be due to using different hyperparameters and embeddings, or I may have missed something in my implementations.

Feel free to feedback if you find something amiss.

## Tested on:

* Tensorflow 1.3.1
* Numpy 1.13.3
* Pything 2.7.1
