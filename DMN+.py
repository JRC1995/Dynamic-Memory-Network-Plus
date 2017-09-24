
# coding: utf-8

# In[1]:


import numpy as np
from __future__ import division

filename = 'glove.6B.50d.txt' 
# (glove data set from: https://nlp.stanford.edu/projects/glove/)


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('GloVe Loaded.')
    file.close()
    return vocab,embd


# Pre-trained GloVe embedding
vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embedding[0]) # word_vec_dim = dimension of each word vectors


# In[2]:


def word2vec(word):  # converts a given word into its vector representation
    return embedding[vocab.index(word)]

def vec2word(vec):   # converts a given vector representation into the represented word 
    for x in xrange(0, len(embedding)):
            if np.array_equal(embedding[x],np.asarray(vec)):
                return vocab[x]


# In[3]:


# example of a vector representation of the word 'example'

print "Vector representation of 'example':\n"
print word2vec("example")


# In[4]:


import string

# Data related to basic induction training and testing from QA bAbi tasks dataset will be used.
# (https://research.fb.com/downloads/babi/)

filename = 'qa16_basic-induction_train.txt' 

fact_story = [] # (fact_story will serve as a list of stories. Each story will be a list of supporting facts. 
                #  Each fact will be a list of word vector representations. It will be a nested list.) 
question = []   #   (A list of questions correspoding to the list of stories. 
                # Each question will be a list of word vector representations) 
answer = []     # (A corresponding list of answers. Each answer will be a single word vector representation.) 


def extract_info(filename):  # extract questions, facts and answers from the file.
    
    fact = [] # will be temporarily filled with a list of supporting facts for a particular story.
              # It will then be fed to fact_story as an item.
    fact_story = [] #same as the globally defined fact_story list object.
    question = []   #same as the globally defined question list object.
    answer = []     #same as the globally defined answer list object.

    file = open(filename,'r')
    
    for line in file.readlines(): # Iterate through one line at a time
        
        flag_end_story = 0 #(flagged as 1 if a story ends)
        line = line.lower() #change all words to lower case
        
        # the line with the question and answer is written at the end of all supporting facts for a specific story
        
        if '?' in line: # checks if we are dealing with a line that includes the question.
            
            flag_end_story=1 # if we are dealing with the line that includes question then it means we have reached
                           # the end of a story
            
            # split the line into question part and the answer part
            
            linesplitindex = line.index('?')
            lineq = line[0:linesplitindex] #lineq represents the question part 
            linea = line[linesplitindex+1:] #linea represents the answer part
            
            # remove punctuations and special characters
            
            lineq = lineq.translate(None, string.punctuation)
            linea = linea.translate(None, string.punctuation)
            
            # the answer part has some '\t' characters where a blank space ' ' is needed. 
            # replacing '\t' with ' ' for future convinience of further splitting the answer part (linea)
            
            linea = linea.replace('\t', ' ')
            
            # split the question part (lineq) into an array of words (rowq). 
            rowq = lineq.strip().split(' ') 
            # split the question part (linea) into an array of words (rowa).
            rowa = linea.strip().split(' ')
            
            embrowq = [] #embrowq will be a list of word vector representations corresponding to the words in rowq
            for i in xrange(1,len(rowq)):
                embrowq.append(word2vec(rowq[i]))
            
            # now embrowq is a list of word vector representations 
            # which represents the words which constitute the question component
            
            question.append(embrowq) # fill the list of questions. 
            
            # the answer should be one word. The answer component also includes 
            # some numbers which serves as the index of the relevant supporting facts.
            # We will ignore those numbers and train the network without the supervising which facts are relevant.
            # Rowa constitutes the list of the word representing the answer and the numbers.
            # We will only take take into account - rowa[0] which is the word that represents the answer.
            
            answer.append(word2vec(rowa[0])) # fill the answer with the vector representation of rowa[0]
                                             # which contains the word that represents the answer to the particular
                                             # question included in this line
            
        else: #else if we are dealing with a supporting fact
            
            # remove punctiations and special characters from the line
            line = line.translate(None, string.punctuation)
            # split the line (a supporting fact) into a list of words
            row = line.strip().split(' ') #row is a list of words in the line (a supporting fact)
            
            embrow = [] # embrow will contain the list of vector representations of correspoding words in row  
            for i in xrange(1,len(row)):
                embrow.append(word2vec(row[i]))
        
            fact.append(embrow) #fill fact with embrow (the list of vector representations of words in a fact)
            
        if flag_end_story == 1: #checks if we have reached the end of a story
            # at the end of a story fact will contain the list of all facts related to the particular story
            fact_story.append(fact)  # fill fact story with 'fact' as an item. fact here is a list of
                                     # supporting facts of a particular story
            fact = [] # resent fact so that it can be filled up with new supporting facts for a new story
            
    file.close()
        
    return fact_story,question,answer

fact_story,question,answer = extract_info(filename)




# In[5]:


from __future__ import division

train_fact_story = []
train_question = []
train_answer = []
val_fact_story = []
val_question = []
val_answer = []

p=90 # p is the train-validation splitting factor i.e. p% of data will be used for training and (100-p)% will be
     # used for validation.
    
train_len = int((p/100)*len(fact_story))
val_len = int(((100-p)/100)*len(fact_story))

train_fact_story = fact_story[0:train_len] #portion of the fact_story that will be used for training
val_fact_story = fact_story[train_len:(train_len+val_len)]#portion of the fact_story that will be used for validation

train_question = question[0:train_len] #portion of questions corresponding to the train_fact_story
val_question = question[train_len:(train_len+val_len)] #portion of questions corresponding to val_fact_story

train_answer = answer[0:train_len] #portion of answers corresponding to train_questions
val_answer = answer[train_len:(train_len+val_len)] #portion of answers corresponding to val_questions.

#Setting up fact_story, question, answer for testing.

test_fact_story = []
test_question = []
test_answer = []

filename = 'qa16_basic-induction_test.txt'

test_fact_story,test_question,test_answer = extract_info(filename)


# In[6]:


#Hyperparameters

# all lines i.e list of word vectors (questions or supporting facts) will be padded will null word vectors 
# so that each of the list representing a line (question or fact) will have the SAME length.
# seq_len represents that length. (Easier to code a RNN in tensorflow when working with a fixed length)

seq_len = 7
hidden_size = 10*word_vec_dim # hidden_size will be the size of representation of encoded facts and questions.
                             # We will use this same hidden_size for all GRUs (besides the final one) 
                             # used in the dynamic memory network
training_iters = 1000 #(epochs)
learning_rate = 0.007
answer_module_timesteps = 10
passes = 30 # passes represents the number of time the memory module will be iterated 


# In[7]:


null_word_vec = np.zeros(word_vec_dim)

#pad facts and questions so that length of each sentence become same.

def padfacts(fact_story):
    pad_stories = []
    
    for i in xrange(0,len(fact_story)):
        pad_facts = []
        
        for j in xrange(0,len(fact_story[i])):
            count = 0
            pad_words = np.zeros([seq_len,word_vec_dim])
            for k in xrange(0,len(fact_story[i][j])):
                pad_words[k]=np.asarray(fact_story[i][j][k])
                count+=1
            for l in xrange(count,seq_len):
                pad_words[l]=null_word_vec #here's where padding takes place
            pad_facts.append(pad_words)
            
        pad_stories.append(np.asarray(pad_facts)) 
        
        # converting pad_facts into numpy array since now each fact is of the same length thanks to padding.
        # it can be later easily converted to a tensor
        
    return pad_stories

def padquestions(question):
    
    pad_questions = []
    
    for i in xrange(0,len(question)):
        pad_words = np.zeros([seq_len,word_vec_dim])
        count = 0
        for j in xrange(0,len(question[i])):
            pad_words[j] = np.asarray(question[i][j])
            count+=1
        for k in xrange(count,seq_len):
            pad_words[k] = null_word_vec #here's where padding takes place
        pad_questions.append(pad_words)
    
    pad_questions = np.asarray(pad_questions)
    
    # converting pad_questions into numpy array since now each question is now of the same length thanks to padding.
    # it can be later easily converted to a tensor
    
    return pad_questions


train_pad_facts = padfacts(train_fact_story)
val_pad_facts = padfacts(val_fact_story)
test_pad_facts = padfacts(test_fact_story)

train_pad_questions = padquestions(train_question)
val_pad_questions = padquestions(val_question)
test_pad_questions = padquestions(test_question)

print "No. of training stories: " + str(len(train_pad_facts))
print "\nShape of the first training story (no. of facts, no. of words in each fact, word vector dimensions):\n"+str(train_pad_facts[0].shape)
print "\nShape of training questions (no. of questions, no. of words in each question, word vector dimensions):\n"+str(train_pad_questions.shape)

# Data preprocessing done.

# To be used for training: (Input: train_pad_facts, train_pad_questions. Target Output: train_answer)
# To be used for validation: (Input: val_pad_facts, val_pad_questions. Target Output: val_answer )
# To be used for testing: (Input: val_)
    


# In[8]:


import tensorflow as tf

# Tensorflow placeholders

# We will feed to the network, one story at a time.
# A story will contain an unknown number of supporting facts, one question, and one answer.

# tf_facts (supporting facts of a story) and tf_question (a question relevant to the story) will serve as Inputs 
# to the network

# tf_fact will be fed with the unknwon number of supporting facts for a specific story.
# Each supporting fact is a nd array of seq_len no. of word_vectors (including padded null_word_vectors).
tf_fact = tf.placeholder(tf.float32, [None,seq_len,word_vec_dim])

# tf_question will be fed with a question for the story. The question should be answerable from the supporting facts.
# The question will be a nd array of seq_len no. of word_vectors (including padded null_word_vectors).
tf_question = tf.placeholder(tf.float32, [seq_len,word_vec_dim])

# tf_answer will serve as the target output.
# tf_answer will be fed with the correct answer for the question. 
# The answer will be a single word vector representation representing one word. 
tf_answer = tf.placeholder(tf.float32,[word_vec_dim])


# In[9]:


"""
The implementation of this model is roughly based on the descriptions presented here:
https://arxiv.org/abs/1603.01417 (henceforth will be referred as DMN+ paper)
Another relevant paper: https://arxiv.org/abs/1506.07285 (the answer module is described here)
"""

# There's already tensorflow library for GRU, but still created my own implementation of a GRU function.

def GRU(x,hprev,wz,uz,bz,wr,ur,br,w,u,bh,t,inp_dim):
    #t = timestep
    i = tf.constant(0,dtype=tf.int32)
    def cond(i,hprev):
        return tf.less(i,t)
    def body(i,hprev):
        inp = tf.reshape(x[i],[1,inp_dim])
        z = tf.sigmoid( tf.matmul(inp,wz) + tf.matmul(hprev,uz) + bz)
        r = tf.sigmoid( tf.matmul(inp,wr) + tf.matmul(hprev,ur) + br)
        h_ = tf.tanh( tf.matmul(inp,w) + tf.multiply(r,tf.matmul(hprev,u)) + bh)
        h = tf.multiply(z,hprev) + tf.multiply((1-z),h_)
        hprev = h
        return i+1,hprev
    i,h = tf.while_loop(cond,body,[i,hprev])
    return h

# custom Attention Based GRU as described in the DMN+ paper. 

def attention_based_GRU(x,hprev,g,wr,ur,br,w,u,bh,t,inp_dim):
    #t=timestep
    i = tf.constant(0,dtype=tf.int32)
    def cond(i,hprev):
        return tf.less(i,t)
    def body(i,hprev):
        inp = tf.reshape(x[i],[1,inp_dim])
        r = tf.sigmoid( tf.matmul(inp,wr) + tf.matmul(hprev,ur) + br)
        h_ = tf.tanh( tf.matmul(inp,w) + tf.multiply(r,tf.matmul(hprev,u)) + bh)
        h = tf.multiply(g[i],hprev) + tf.multiply((1-g[i]),h_)
        hprev = h
        return i+1,hprev
    i,h = tf.while_loop(cond,body,[i,hprev])
    return h

# The overall DMN+ model

def DMN_Plus_Model(tf_facts,tf_question):
   
    #input module (encodes facts - create representations of eachfacts)
    
    #Implementation of Input Fusion Layer
    
    #initialization of input module parameters for first layer of GRU (forward GRU)
    
    wzimf = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uzimf = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wrimf = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    urimf = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wimf = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uimf = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    bzimf = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    brimf = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bhimf = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    
    #initialization of input module parameters for second layer of GRU (backward GRU)
    
    wzimb = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uzimb = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wrimb = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    urimb = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wimb = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uimb = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    bzimb = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    brimb = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bhimb = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    
    
    facts_shape = tf.shape(tf_facts)
    facts_num = facts_shape[0] #no. of facts (fact no. facts_num-1 is the last fact starting from fact no. 0)
    
    #forward GRU
    
    i = tf.constant(0) #i = 0. Loop will start from the first fact - fact no. 0.
    hprev = tf.zeros([1,hidden_size],tf.float32) # Initialization hidden layer state
    srf = tf.TensorArray(dtype=tf.float32,size=facts_num) # Initialization of srf
                                                          # srf = forward representations of supporting facts
        
    def condf(i,hprev,srf):
        return i < facts_num
    
    def bodyf(i,hprev,srf):
        
        inp = tf_facts[i] #inp is an array of word_vecs representating fact no. i.
        hprev = GRU(inp,hprev,wzimf,uzimf,bzimf,wrimf,urimf,brimf,wimf,uimf,bhimf,seq_len,word_vec_dim)
                # seq_len is the timestep for the GRU. Seq_len = no. of words in fact_no. i. 
                # The GRU will loop through one word vector at a time giving the final hidden state at the end.
                # The final hidden state (the new resultant hprev) will be the forward representation of fact i.
        srf = srf.write(i,tf.reshape(hprev,[hidden_size])) #srf index i will contain
                                                           #the forward fact representation of fact i.
        return i+1,hprev,srf #the resultant hprev from fact i will be fed as the inital hidden state to the GRU
                             #in the next loop (going forward) whic will deal with fact i+1. That is, fact i+1 
                             #will be encoded 
                             #in the CONTEXT of fact i which had been encoded in the context of previous facts
                             #(fact i-1), if any
    
    i,hprev,srf = tf.while_loop(condf,bodyf,[i,hprev,srf]) 
    
    
    #backward GRU
    
    i = facts_num
    i = i-1 #here i becomes facts_num-1. Loop will start from the last fact - fact no. fact_num-1 
    
    srb = tf.TensorArray(dtype=tf.float32,size=facts_num) # Initialization of srb
                                                          # srb = backward representations of supporting facts
    hprev = tf.zeros([1,hidden_size],tf.float32) #initialization of hidden layer state.
        
    def condb(i,hprev,srb):
        return i >= 0
    
    def bodyb(i,hprev,srb):
        
        inp = tf_facts[i] #inp is an array of word_vecs representating fact no. i.
        hprev = GRU(inp,hprev,wzimb,uzimb,bzimb,wrimb,urimb,brimb,wimb,uimb,bhimb,seq_len,word_vec_dim)
                # seq_len is the timestep for the GRU. Seq_len = no. of words in fact_no. i. 
                # The GRU will loop through one word vector at a time giving the final hidden state at the end.
                # The final hidden state (the new resultant hprev) will be the backward representation of fact i.
        srb = srb.write(i,tf.reshape(hprev,[hidden_size])) #srb index i will contain
                                                           #the backward fact representation of fact i.
        return i-1,hprev,srb #the resultant hprev from fact i will be fed as the inital hidden state to the GRU
                             #in the next loop (going bacward) which will deal with fact i-1. That is, fact i-1 
                             #will be encoded 
                             #in the CONTEXT of fact i which had been encoded in the context of later facts
                             #(fact i+1), if any
    
    i,hprev,srb = tf.while_loop(condb,bodyb,[i,hprev,srb])

        
    #fusion
    
    fusion = srf.stack() + srb.stack() #srf.stack() will make a tensor out of srf values.
                                       #Same for srb.gather(indices)
    #fusion = list of final representation of supporting facts - result of fusing srb and srf
    
    fusion_len = facts_num
    
    
    
    #initialization of question module parameters
    
    wzqm = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uzqm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wrqm = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    urqm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wqm = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=5e-2))
    uqm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    bzqm = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    brqm = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bhqm = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    
    hprev = tf.zeros([1,hidden_size],tf.float32) #initialization of hidden layer state for 
                                                 #the GRU to be used for question representation
    
   
    qr = GRU(tf_question,hprev,wzqm,uzqm,bzqm,wrqm,urqm,brqm,wqm,uqm,bhqm,seq_len,word_vec_dim)
    # qr = question representation (formed by encoding word vectors in tf_question through a GRU)
    
    
    #initialization of attention based gru module
    
    wrattm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    urattm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wattm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    uattm = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    brattm = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bhattm = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    
    
    #initialization of memory update parameters
    
    wt = tf.Variable(tf.truncated_normal(shape=[passes,hidden_size*3,hidden_size],stddev=5e-2))
    bt = tf.Variable(tf.truncated_normal(shape=[passes,1,hidden_size],stddev=5e-2))

    mprev = qr #initialization of initial memory state (m0)

    inter_neurons = 100 #number of hidden layer neurons for some portions of a neural network.
    
    #parameters required for calculating single scalar value g (attention gate)
    
    w1 = tf.Variable(tf.truncated_normal(shape=[hidden_size*4,inter_neurons],stddev=5e-2))
    b1 = tf.Variable(tf.truncated_normal(shape=[inter_neurons],stddev=5e-2))
    w2 = tf.Variable(tf.truncated_normal(shape=[inter_neurons,1],stddev=5e-2))
    b2 = tf.Variable(tf.truncated_normal(shape=[1],stddev=5e-2))
    
    i=tf.constant(0)
    
    def attcond(i,mprev):
        return i<passes
    
    def attbody(i,mprev):
        
        #Implementation of the equations for computing the attention gate (g) value from the DMN+ paper. 
        
        zc1 = tf.multiply(fusion,qr)
        zc2 = tf.multiply(fusion,mprev)
        zc3 = tf.abs(tf.subtract(fusion,qr))
        zc4 = tf.abs(tf.subtract(fusion,mprev))
        z = tf.concat([zc1,zc2,zc3,zc4],1)
        capZ = tf.add( tf.matmul( tf.tanh( tf.add( tf.matmul(z,w1),b1 ) ),w2 ) , b2)
        
        #There should be one single scalar g score for each fact-representation.
        #Here g is a list of g-scores corresponding to the list of fact-representations in fusion.
        
        g = tf.nn.softmax(capZ)
        
        
        #soft attention (following the DMN+ paper)
        
        # c is the contextual vector which is produced as a result of weighted summation of fact-representations
        # and their corresponding g score
        # (Summation (i = 0 to fact_num-1) g[i]*fusion[i])
        
        c = tf.reduce_sum(tf.multiply(fusion,g),0)
        c = tf.reshape(c,[1,hidden_size])
        
        #attention based GRU (modified GRU - uses g(attention gate) instead of z(update gate))
        
        mprev = attention_based_GRU(fusion,mprev,g,wrattm,urattm,brattm,wattm,uattm,bhattm,fusion_len,hidden_size)
            
        #memory episode update following DMN+ paper
        
        mprev = tf.nn.relu(tf.matmul(tf.concat([mprev,c,qr],1),wt[i]) + bt[i])
        
        return i+1,mprev #returns final updated memory state mprev to the next iteration\pass.
                         #In each pass the network should find deeper information from the supporting facts.
                         #We are using no. of passes as a hyperparameter. 
    
    i,mprev = tf.while_loop(attcond,attbody,[i,mprev]) 
    
    #Answer module - the final module that computes the answer.  
    
    wa1 = tf.Variable(tf.truncated_normal(shape=[hidden_size,word_vec_dim],stddev=5e-2))
    
    #initialization of answer module GRU parameters
    
    wza = tf.Variable(tf.truncated_normal(shape=[(word_vec_dim+hidden_size),hidden_size],stddev=5e-2))
    uza = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wra = tf.Variable(tf.truncated_normal(shape=[(word_vec_dim+hidden_size),hidden_size],stddev=5e-2))
    ura = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    wa = tf.Variable(tf.truncated_normal(shape=[(word_vec_dim+hidden_size),hidden_size],stddev=5e-2))
    ua = tf.Variable(tf.truncated_normal(shape=[hidden_size,hidden_size],stddev=5e-2))
    bza = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bra = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    bha = tf.Variable(tf.truncated_normal(shape=[1,hidden_size],stddev=5e-2))
    
    #answer module GRU 
        
    aprev=mprev #initializes the hidden state of answer module GRU with final memory state computed in the previous
                #module
    
    timesteps = tf.constant(answer_module_timesteps)
    i = tf.constant(0)
    
    def condam(i,aprev):
        return i<timesteps
    def bodyam(i,aprev):
        
        #implementation of the equations of answer module as presented in (https://arxiv.org/abs/1506.07285)
        
        yprev = tf.nn.softmax(tf.matmul(aprev,wa1))
        concat = tf.concat([yprev,qr],1)
        t = 1 #t or timsteps for the following GRU is one since concat only one
              #(word_vec_dim+hidden_szie) dimensional input will be fed. We will not be iterating through
              #multiple words - creating hidden states in the context of previous words.
              #However we are already running the whole GRU function under a specified timestep (tf.constant timesteps), 
              #and in each timestep new input is created and previous aprev is used as the new initial aprev
              #for the GRU. So we don't further need t to be more than 1.
        aprev = GRU(concat,aprev,wza,uza,bza,wra,ura,bra,wa,ua,bha,t,(hidden_size+word_vec_dim))
        return i+1,aprev
    
    i,aprev = tf.while_loop(condam,bodyam,[i,aprev])
        
    y = tf.matmul(aprev,wa1) #the final predicted answer
    y = tf.reshape(y,[word_vec_dim])    
    return y


# In[10]:


def nearest_neighbour(x,y): #Returns the tensor in x which is closest (in terms of Euclidean distance)
    s=tf.subtract(x,y)      #to the tensor y. 
    s=tf.square(s)
    r=tf.reduce_sum(s,1)
    r=tf.sqrt(r)
    return x[tf.argmin(r)]


# Construct model
model_output = DMN_Plus_Model(tf_fact,tf_question)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=tf.nn.softmax(tf_answer)))

global_step = tf.Variable(0)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)


#Evaluate model
correct_pred = tf.equal(nearest_neighbour(tf.convert_to_tensor(embedding),model_output),tf_answer)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = nearest_neighbour(tf.convert_to_tensor(embedding),model_output)

# Initializing the variables
init = tf.global_variables_initializer()


# In[ ]:


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 1   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_acc=0
    
    while step <= training_iters:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0
        
        
        for i in xrange(0,len(train_pad_facts)):
            
            # Run optimization operation (backpropagation)
            _,loss,acc,pred = sess.run([optimizer,cost,accuracy,prediction],feed_dict={tf_fact: train_pad_facts[i], 
                                                                       tf_question: train_pad_questions[i], 
                                                                       tf_answer: train_answer[i]})
        
            total_loss += loss
            total_acc += acc
                
            if i%100 == 0:
                print "Iter "+str(i)+", Loss= "+                      "{:.3f}".format(loss)+", Predicted Answer= "+                        str(vec2word(pred))+", Actual Answer= "+                        str(vec2word(train_answer[i]))
                        
        avg_loss = total_loss/len(train_pad_facts) # Average training loss
        avg_acc = total_acc/len(train_pad_facts)  # Average training accuracy
        
        loss_list.append(avg_loss) # Storing values in list for plotting later on.
        acc_list.append(avg_acc) # Storing values in list for plotting later on.

        for i in xrange(0,len(val_pad_facts)):
            
            val_loss, val_acc = sess.run([cost, accuracy], feed_dict={tf_fact: val_pad_facts[i], 
                                                                      tf_question: val_pad_questions[i], 
                                                                      tf_answer: val_answer[i]})
            total_val_loss += val_loss
            total_val_acc += val_acc
                      
            
        avg_val_loss = total_val_loss/len(val_pad_facts) # Average validation loss
        avg_val_acc = total_val_acc/len(val_pad_facts) # Average validation accuracy
             
        val_loss_list.append(avg_val_loss) # Storing values in list for plotting later on.
        val_acc_list.append(avg_val_acc) # Storing values in list for plotting later on.
    

        print "\nIter " + str(step) + ", Validation Loss= " +                 "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " +                 "{:.3f}%".format(avg_val_acc*100)+""
        print "Iter " + str(step) + ", Average Training Loss= " +               "{:.3f}".format(avg_loss) + ", Average Training Accuracy= " +               "{:.3f}%".format(avg_acc*100)+""
                    
        if avg_val_acc > best_val_acc: # When better accuracy is received than previous best validation accuracy
                
            best_val_acc = avg_val_acc # update value of best validation accuracy received yet.
            saver.save(sess, 'Model_Backup/model.ckpt') # save_model including model variables (weights, biases etc.)
            print "Checkpoint created!"

            
        step += 1
        
    print "\nOptimization Finished!\n"
    
    print "Best Validation Accuracy: %.3f%%"%((best_val_acc)*100)
    
    #The model can be run on test data set after this.
    #val_loss_list, val_acc_list, loss_list and acc_list can be used for plotting. 
    


# In[ ]:




