# Implementation of Dynamic Memory Network+   

Implementation of Dynamic Memory Network+ (for question answering) using Tensorflow.

The implementation is based on the model proposed in 

["Dynamic Memory Networks for Visual and Textual Question Answering" 
by Caiming Xiong, Stephen Merity, Richard Socher, arXiv:1603.01417](https://arxiv.org/abs/1603.01417). 

The original Dynamic Memory Network was introduced in 

["Ask Me Anything: Dynamic Memory Networks for Natural Language Processing" 
by Ankit Kumar, Peter Ondruska, Mohit Iyyer, James Bradbury, Ishaan Gulrajani, Victor Zhong,Romain Paulus, Richard Socher, arXiv:1506.07285](https://arxiv.org/pdf/1506.07285.pdf) 

(I had to refer to this paper too).

This DMN+ Model uses:

* Word vectors in facts are positionally encoded, and added to create sentence representations.
* Bi-directional GRU is used over the sentence representations in the funsion layer. The forward and backward list of hidden states are added.
* Attention Based GRU is used in the episodic memory module.
* A linear layer with ReLu activation is used along with untied weights to update the memory for the next pass. 

I also included layer normalization ([Layer Normalization - Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton](https://arxiv.org/abs/1607.06450)) before every activation, barring the pre-activation state of the final layer. 

I used pre-trained GloVe embedding downloaded from [here](https://nlp.stanford.edu/projects/glove/).
I used the 100 dimensional embeddings. 

I trained the model on basic induction tasks from [bAbi-tasks dataset](https://research.fb.com/downloads/babi/). 

Hyperparameters are different from the original implementation.

## Hyperparameters used:

* Hidden size = 100
* Embedding dimensions = 100
* Learning rate = 0.001
* Passes = 3
* Mini Batch Size = 128
* L2 Regularization = 0.0001
* Dropout Rate = 0.1
* Initialization = 0 for biases, Xavier for weights

(last 10% of data samples used for validation.)

## Result Discussion: 

I trained the model in a weakly supervised fashion. That is, the model won't be told which supporting facts are relevant for inductive reasoning in order to derive an answer. 

The network starts to overfit around the 35th epoch. The validation cost starts to increase, while the training cost keeps on decreasing. 

The published classification error of QA task 16 (basic induction) of bAbi Dataset of the DMN+ model (as given here: https://arxiv.org/pdf/1603.01417.pdf - page 7) is 45.3. 

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

My implementation of the model on pretrained 100 dimensional GloVe vectors seems to produce about **51% classification accuracy**  on Test Data for induction tasks (check DMN+.ipynb)...i.e the **classification error is 49%**. . 

The error is less than what the original DMN model acheived (error 55.1%) as specified in the paper, but still greater than the errors achieved achieved by the original implementation of the improved versions of DMN (DMN1, DMN2, DMN3, DMN+) in the paper.

This could be due to using different hyperparameters and embeddings, or I may have missed something in my implementations.

## File Descriptions:

**QA_PreProcess.py\QA_PreProcess.ipynb:** Converts the raw induction tasks data set to separate ndarrays containing questions, answers, and facts with all words being in the form of GloVe pre-trained vector representations.  

**DMN+.py\DMN+.ipynb:** The DMN+ model, along with training, validation and testing. 

## Tested on:

* Tensorflow 1.4 
* Numpy 1.13.3
* Python 2.7.12
