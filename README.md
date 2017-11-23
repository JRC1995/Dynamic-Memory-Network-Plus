# Implementation of Dynamic Memory Network + 

Implementation of Dynamic Memory Network + (for question answering) using Tensorflow.

The model is based on this [paper](https://arxiv.org/abs/1603.01417). The original Dynamic Memory Network is introduced in this [paper](https://arxiv.org/pdf/1506.07285.pdf) (I had to refer to this paper too).

This is the DMN+ model as mentioned in the paper. It uses:

* Word vectors in facts are positionally encoded, and added to create sentence representations.
* Bi-directional GRU is used over the sentence representations in the funsion layer. The forward and backward list of hidden states are added.
* Attention Based GRU is used in the episodic memory module.
* A linear layer with ReLu activation is used along with untied weights to update the memory for the next pass. 

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).
I used the 100 dimensional embeddings. 

I trained the model on basic induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

Hyperparameters are slightly different from the original implementation.

## Hyperparameters used:

* Hidden size = 100
* Embedding dimensions = 100
* Learning rate = 0.001
* Passes = 3
* Mini Batch Size = 128
* L2 Regularization = 0.0001
* Dropout Rate = 0.1
* Initialization = Xavier

(last 10% of data samples used for validation.)

## Result Discussion: 

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

## File Descriptiosn:

**QA_PreProcess.py\QA_PreProcess.ipynb:** Converts the raw induction tasks data set to separate ndarrays containing questions, answers, and facts with all words being in the form of GloVe pre-trained vector representations.  

**DMN+.py\DMN+.ipynb:** The DMN+ model where the single word answer is computed from a probability distribution which is computed by one linear layer transforming the final memory state of the episodic memory module. 

**DMN+ GRU_answer.py\DMN+ GRU_answer.ipynb:** The DMN+ model where the single word answer is computedin two step. In the first step, a GRU is used for one timestep with the question representation as the input and the final memory state of the episodic memory module as the initial hidden state. In the second step, the output of the GRU is linearly transformed to a probability distribution, from which the final answer is dervied (the one with the maximum probability).

## Tested on:

* Tensorflow 1.3.1
* Numpy 1.13.3
* Pything 2.7.1
