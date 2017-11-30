
# coding: utf-8

# ### LOADING PREPROCESSED DATA
# 
# Loading GloVe word embeddings. Building functions to convert words into their vector representations and vice versa. Loading babi induction task 10K dataset.

# In[1]:


import numpy as np
from __future__ import division

filename = 'glove.6B.100d.txt'

def loadEmbeddings(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded!')
    file.close()
    return vocab,embd
vocab,embd = loadEmbeddings(filename)


word_vec_dim = len(embd[0])

vocab.append('<UNK>')
embd.append(np.asarray(embd[vocab.index('unk')],np.float32)+0.01)

vocab.append('<EOS>')
embd.append(np.asarray(embd[vocab.index('eos')],np.float32)+0.01)

vocab.append('<PAD>')
embd.append(np.zeros((word_vec_dim),np.float32))

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('<UNK>')]

def most_similar_eucli(x):
    xminusy = np.subtract(embedding,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)

def vec2word(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli(np.asarray(vec,np.float32))
    return vocab[most_similars[0]]

import pickle

with open ('positionalPICKLE', 'rb') as fp:
    processed_data = pickle.load(fp)

fact_stories = processed_data[0]
questions = processed_data[1]
answers = np.reshape(processed_data[2],(len(processed_data[2])))
test_fact_stories = processed_data[3]
test_questions = processed_data[4]
test_answers = np.reshape(processed_data[5],(len(processed_data[5])))


# In[2]:


import random

print "EXAMPLE DATA:\n"

sample = random.randint(0,len(fact_stories))

print "FACTS:\n"
for i in xrange(len(fact_stories[sample])):
    print str(i+1)+") ",
    print map(vec2word,fact_stories[sample][i])
    
print "\nQUESTION:"
print map(vec2word,questions[sample])
print "\nANSWER:"
print vocab[answers[sample]]


# ### CREATING TRAINING AND VALIDATION DATA

# In[3]:


from __future__ import division

train_fact_stories = []
train_questions = []
train_answers = []
val_fact_stories = []
val_questions = []
val_answers = []

p=90 #(90% data used for training. Rest for validation)
    
train_len = int((p/100)*len(fact_stories))
val_len = int(((100-p)/100)*len(fact_stories))

train_fact_stories = fact_stories[0:train_len] 
val_fact_stories = fact_stories[train_len:(train_len+val_len)]

train_questions = questions[0:train_len] 
val_questions = questions[train_len:(train_len+val_len)] 

train_answers = answers[0:train_len] 
val_answers = answers[train_len:(train_len+val_len)] 


# ### Function to create randomized batches

# In[4]:


def create_batches(fact_stories,questions,answers,batch_size):
    
    shuffle = np.arange(len(questions))
    np.random.shuffle(shuffle)
    
    batches_fact_stories = []
    batches_questions = []
    batches_answers = []
    
    i=0
    
    while i+batch_size<=len(questions):
        batch_fact_stories = []
        batch_questions = []
        batch_answers = []
        
        for j in xrange(i,i+batch_size):
            batch_fact_stories.append(fact_stories[shuffle[j]])
            batch_questions.append(questions[shuffle[j]])
            batch_answers.append(answers[shuffle[j]])
            
        batch_fact_stories = np.asarray(batch_fact_stories,np.float32)

        batch_questions = np.asarray(batch_questions,np.float32)
        
        batches_fact_stories.append(batch_fact_stories)
        batches_questions.append(batch_questions)
        batches_answers.append(batch_answers)
        
        i+=batch_size
        
    batches_fact_stories = np.asarray(batches_fact_stories,np.float32)
    batches_questions = np.asarray(batches_questions,np.float32)
    batches_answers = np.asarray(batches_answers,np.float32)
    
    return batches_fact_stories,batches_questions,batches_answers
    


# ### Hyperparameters

# In[5]:


import tensorflow as tf

# Tensorflow placeholders

tf_facts = tf.placeholder(tf.float32, [None,None,None,word_vec_dim])
tf_questions = tf.placeholder(tf.float32, [None,None,word_vec_dim])
tf_answers = tf.placeholder(tf.int32,[None])
keep_prob = tf.placeholder(tf.float32)

#hyperparameters
epochs = 256
learning_rate = 0.001
hidden_size = 100
beta = 0.001 #l2 regularization scale
regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
heads = 8
h = heads
dqkv = 32


# ### Parameters

# In[6]:



init = tf.zeros_initializer()

with tf.variable_scope("question_encoding"):
    
    # FORWARD GRU PARAMETERS
    
    wzf = tf.get_variable("wzf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer= regularizer)
    uzf = tf.get_variable("uzf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzf = tf.get_variable("bzf", shape=[hidden_size],initializer=init)

    wrf = tf.get_variable("wrf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urf = tf.get_variable("urf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brf = tf.get_variable("brf", shape=[hidden_size],initializer=init)

    wf = tf.get_variable("wf", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    uf = tf.get_variable("uf", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bf = tf.get_variable("bf", shape=[hidden_size],initializer=init)

    # BACKWARD GRU PARAMETERS

    wzb = tf.get_variable("wzb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    uzb = tf.get_variable("uzb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzb = tf.get_variable("bzb", shape=[hidden_size],initializer=init)

    wrb = tf.get_variable("wrb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urb = tf.get_variable("urb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brb = tf.get_variable("brb", shape=[hidden_size],initializer=init)

    wb = tf.get_variable("wb", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    ub = tf.get_variable("ub", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bb = tf.get_variable("bb", shape=[hidden_size],initializer=init)
    

with tf.variable_scope("facts_word_encoding"):
    
    # FORWARD GRU PARAMETERS
    
    wzf = tf.get_variable("wzf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer= regularizer)
    uzf = tf.get_variable("uzf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzf = tf.get_variable("bzf", shape=[hidden_size],initializer=init)

    wrf = tf.get_variable("wrf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urf = tf.get_variable("urf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brf = tf.get_variable("brf", shape=[hidden_size],initializer=init)

    wf = tf.get_variable("wf", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    uf = tf.get_variable("uf", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bf = tf.get_variable("bf", shape=[hidden_size],initializer=init)

    # BACKWARD GRU PARAMETERS

    wzb = tf.get_variable("wzb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    uzb = tf.get_variable("uzb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzb = tf.get_variable("bzb", shape=[hidden_size],initializer=init)

    wrb = tf.get_variable("wrb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urb = tf.get_variable("urb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brb = tf.get_variable("brb", shape=[hidden_size],initializer=init)

    wb = tf.get_variable("wb", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    ub = tf.get_variable("ub", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bb = tf.get_variable("bb", shape=[hidden_size],initializer=init)

    
with tf.variable_scope("facts_sentence_embedding_layer1"):
    
    Wq = tf.get_variable("Wq", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wk = tf.get_variable("Wk", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wv = tf.get_variable("Wv", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wo = tf.get_variable("Wo", shape=[h*dqkv,hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    
with tf.variable_scope("facts_sentence_embedding_layer2"):
    
    Wq = tf.get_variable("Wq", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wk = tf.get_variable("Wk", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wv = tf.get_variable("Wv", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wo = tf.get_variable("Wo", shape=[h*dqkv,hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
       
with tf.variable_scope("facts_sentence_encoding"):
    
    # FORWARD GRU PARAMETERS
    
    wzf = tf.get_variable("wzf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer= regularizer)
    uzf = tf.get_variable("uzf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzf = tf.get_variable("bzf", shape=[hidden_size],initializer=init)

    wrf = tf.get_variable("wrf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urf = tf.get_variable("urf", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brf = tf.get_variable("brf", shape=[hidden_size],initializer=init)

    wf = tf.get_variable("wf", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    uf = tf.get_variable("uf", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bf = tf.get_variable("bf", shape=[hidden_size],initializer=init)

    # BACKWARD GRU PARAMETERS

    wzb = tf.get_variable("wzb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    uzb = tf.get_variable("uzb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    bzb = tf.get_variable("bzb", shape=[hidden_size],initializer=init)

    wrb = tf.get_variable("wrb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    urb = tf.get_variable("urb", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
    brb = tf.get_variable("brb", shape=[hidden_size],initializer=init)

    wb = tf.get_variable("wb", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    ub = tf.get_variable("ub", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
    bb = tf.get_variable("bb", shape=[hidden_size],initializer=init)
    
    
with tf.variable_scope("answer_layer1"):
    
    Wq = tf.get_variable("Wq", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wk = tf.get_variable("Wk", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wv = tf.get_variable("Wv", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wo = tf.get_variable("Wo", shape=[h*dqkv,hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    
with tf.variable_scope("answer_layer2"):
    
    Wq = tf.get_variable("Wq", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wk = tf.get_variable("Wk", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wv = tf.get_variable("Wv", shape=[h,hidden_size,dqkv],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)
    Wo = tf.get_variable("Wo", shape=[h*dqkv,hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)

wpd = tf.get_variable("wpd", shape=[hidden_size,len(vocab)],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer= regularizer)


# ### Layer Normalization for NxH shaped inputs

# In[7]:


def layer_norm_2d(inputs,scope,epsilon = 1e-5):
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        scale = tf.get_variable("scale", shape=[inputs.get_shape()[1]],
                        initializer=tf.ones_initializer())
        shift = tf.get_variable("shift", shape=[inputs.get_shape()[1]],
                        initializer=tf.zeros_initializer())
        
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)),(inputs - mean)) + shift
 
    return LN


# ### Layer Normalization for NxMxH shaped inputs

# In[8]:


def layer_norm_3d(inputs,scope,epsilon = 1e-5):
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        scale = tf.get_variable("scale", shape=[1,1,inputs.get_shape()[2]],
                        initializer=tf.ones_initializer())
        shift = tf.get_variable("shift", shape=[1,1,inputs.get_shape()[2]],
                        initializer=tf.zeros_initializer())
        
    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)),(inputs - mean)) + shift
 
    return LN


# ### Multi-Headed Attention

# In[9]:



def attention(Q,K,V):

    d = tf.cast(dqkv,tf.float32)
    K = tf.transpose(K,[0,2,1])
    
    softmax_component = tf.div(tf.matmul(Q,K),tf.sqrt(d))

    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component),keep_prob),V)
 
    return result
       

def multihead_attention(Q,K,V,scope):
    
    d = dqkv
    
    Q_ = tf.reshape(Q,[-1,tf.shape(Q)[2]])
    K_ = tf.reshape(K,[-1,tf.shape(Q)[2]])
    V_ = tf.reshape(V,[-1,tf.shape(Q)[2]])

    heads = tf.TensorArray(size=h,dtype=tf.float32)
    
    with tf.variable_scope(scope, reuse=True):
        
        Wq = tf.get_variable('Wq')
        Wk = tf.get_variable('Wk')
        Wv = tf.get_variable('Wv')
        Wo = tf.get_variable('Wo')
    
    for i in xrange(0,h):
        
        Q_w = tf.matmul(Q_,Wq[i])
        Q_w = tf.reshape(Q_w,[tf.shape(Q)[0],tf.shape(Q)[1],d])
        
        K_w = tf.matmul(K_,Wk[i])
        K_w = tf.reshape(K_w,[tf.shape(K)[0],tf.shape(K)[1],d])
        
        V_w = tf.matmul(V_,Wv[i])
        V_w = tf.reshape(V_w,[tf.shape(V)[0],tf.shape(V)[1],d])

        head = attention(Q_w,K_w,V_w)
            
        heads = heads.write(i,head)
        
    heads = heads.stack()
    
    concated = heads[0]
    
    for i in xrange(1,h):
        concated = tf.concat([concated,heads[i]],2)

    concated = tf.reshape(concated,[-1,h*d])
    out = tf.matmul(concated,Wo)
    out = tf.reshape(out,[tf.shape(heads)[1],tf.shape(heads)[2],word_vec_dim])
    
    return out
    


# ### Bi-Directional GRU
# 
# Returns summation of backward and forward hidden states. 

# In[10]:


def bi_GRU(inp,hidden,seq_len,scope):
    
    #inp shape = batch_size x seq_len x vector_dimension
    
    inp = tf.transpose(inp,[1,0,2])
    
    #now inp shape = seq_len x batch_size x vector_dimension
    
    hidden_forward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    hidden_backward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hiddenf = hidden
    hiddenb = hidden

    with tf.variable_scope(scope, reuse=True):
        
        wzf = tf.get_variable("wzf")
        uzf = tf.get_variable("uzf")
        bzf = tf.get_variable("bzf")
        
        wrf = tf.get_variable("wrf")
        urf = tf.get_variable("urf")
        brf = tf.get_variable("brf")
        
        wf = tf.get_variable("wf")
        uf = tf.get_variable("uf")
        bf = tf.get_variable("bf")
        
        wzb = tf.get_variable("wzb")
        uzb = tf.get_variable("uzb")
        bzb = tf.get_variable("bzb")
        
        wrb = tf.get_variable("wrb")
        urb = tf.get_variable("urb")
        brb = tf.get_variable("brb")
        
        wb = tf.get_variable("wb")
        ub = tf.get_variable("ub")
        bb = tf.get_variable("bb")
        
    i = 0
    j = seq_len - 1
    
    def cond(i,j,hiddenf,hiddenb,hidden_forward,hidden_backward):
        return i < seq_len
    
    def body(i,j,hiddenf,hiddenb,hidden_forward,hidden_backward):
        
        xf = inp[i]
        xb = inp[j]

        # FORWARD GRU EQUATIONS:
        z = tf.sigmoid( layer_norm_2d( tf.matmul(xf,wzf) + tf.matmul(hiddenf,uzf) + bzf, scope))
        r = tf.sigmoid( layer_norm_2d( tf.matmul(xf,wrf) + tf.matmul(hiddenf,urf) + brf, scope))
        h_ = tf.tanh( layer_norm_2d( tf.matmul(xf,wf) + tf.multiply(r,tf.matmul(hiddenf,uf)) + bf, scope))
        hiddenf = tf.multiply(z,h_) + tf.multiply((1-z),hiddenf)

        hidden_forward = hidden_forward.write(i,hiddenf)
        
        # BACKWARD GRU EQUATIONS:
        z = tf.sigmoid( tf.matmul(xb,wzb) + tf.matmul(hiddenb,uzb) + bzb )
        r = tf.sigmoid( tf.matmul(xb,wrb) + tf.matmul(hiddenb,urb) + brb )
        h_ = tf.tanh( tf.matmul(xb,wb) + tf.multiply(r,tf.matmul(hiddenb,ub)) + bb )
        hiddenb = tf.multiply(z,h_) + tf.multiply((1-z),hiddenb)
        
        hidden_backward = hidden_backward.write(j,hiddenb)
        
        
        return i+1,j-1,hiddenf,hiddenb,hidden_forward,hidden_backward
    
    _,_,_,_,hidden_forward,hidden_backward = tf.while_loop(cond,body,[i,j,
                                                                        hiddenf,
                                                                        hiddenb,
                                                                        hidden_forward,
                                                                        hidden_backward])
    
    forward = hidden_forward.stack()
    backward = hidden_backward.stack()
    
    hidden_list = forward + backward
    
    #forward\backward\hidden_list shape = seq_len x  batch_size x hidden_size
    
    return hidden_list
    

        


# ### Custom transformer based Attenton Mechanism

# In[11]:


def multi_attend(fact,qr,scope):

    # SUBLAYER 1 (MULTI HEADED SELF ATTENTION)

    sublayer1 = multihead_attention(fact,fact,fact,scope+"_layer1")
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm_3d(sublayer1 + fact,scope+"_layer1")
    
    # SUBLAYER 2 (MULTI HEADED QUESTION-BASED ATTENTION)
    
    sublayer2 = multihead_attention(qr,sublayer1,sublayer1,scope+"_layer2")
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm_3d(sublayer2,scope+"_layer2")
    
    return sublayer2


# ### Custom Model 

# In[ ]:


def model(tf_facts,tf_questions):
    
    tf_batch_size = tf.shape(tf_facts)[0]
    facts_num = tf.shape(tf_facts)[1]
    fact_len = tf.shape(tf_facts)[2]
    
    question_len = tf.shape(tf_questions)[1]
    
    hidden = tf.zeros([tf_batch_size,hidden_size],tf.float32)
    
    # word level question encoding
    qr = bi_GRU(tf_questions,hidden,question_len,scope="question_encoding")
    
    #now qr shape = question_len x batch_size x hidden_size
    
    # question representation (qr) = average of the GRU bi-hidden states
    qr = tf.reduce_mean(qr,0)
    
    # now qr shape = batch_size x hidden_size
    
    qr = tf.reshape(qr,[tf_batch_size,1,hidden_size])
    
    # Encoding words in facts
    
    tf_facts = tf.reshape(tf_facts,[tf_batch_size,facts_num*fact_len,word_vec_dim])
    
    tf_facts = bi_GRU(tf_facts,tf.reshape(qr,[tf_batch_size,hidden_size]),
                      facts_num*fact_len,
                      scope="facts_word_encoding")
    
    # now tf_facts shape = facts_num*fact_len x batch_size x hidden_size
    
    tf_facts = tf.transpose(tf_facts,[1,0,2])
    
    # now tf_facts shape = batch_size x facts_num*fact_len x hidden_size
    
    tf_facts = tf.reshape(tf_facts,[tf_batch_size,facts_num,fact_len,hidden_size])

    tf_facts = tf.transpose(tf_facts,[1,0,2,3])
    
    # now tf_facts shape = facts_num x batch_size x fact_len x hidden_size
    
    # Heirarchical attention with transformers with question as key
    # And sentence embedding
                            
    i = 0
    facts_list = tf.TensorArray(size=facts_num,dtype=tf.float32)
    
    def cond(i,facts_list):
        return i < facts_num
    
    def body(i,facts_list):
        
        attended_fact = multi_attend(tf_facts[i],qr,"facts_sentence_embedding")
        
        # attended_fact shape = batch_size x 1 x hidden_size
        
        facts_list = facts_list.write(i,attended_fact)
             
        return i+1,facts_list
             
    _,facts_list = tf.while_loop(cond,body,[i,facts_list])
             
    facts_list = facts_list.stack()
    
    # now facts_list shape = facts_num x batch_size x 1 x hidden_size
    
    facts_list = tf.reshape(facts_list,[facts_num,tf_batch_size,hidden_size])
    
    # sentence encoding
    
    facts_list = tf.transpose(facts_list,[1,0,2])
    
    # now facts_list shape = batch_size x facts_num x hidden_size
    
    facts_list = bi_GRU(facts_list,tf.reshape(qr,[tf_batch_size,hidden_size]),
                        facts_num,
                        scope="facts_sentence_encoding")
    
    # now facts_list shape = facts_num x batch_size x hidden_size
    
    facts_list = tf.transpose(facts_list,[1,0,2])
    
    # now facts_list shape = batch_size x facts_num x hidden_size
     
    # Answer Module

    y = multi_attend(facts_list,qr,"answer")
    
    y = tf.reshape(y,[tf_batch_size,hidden_size])

    # y shape = batch_size x hidden_size
    
    y = tf.matmul(y,wpd)
                            
    return y 
    


# ### Regularization, Cost Function, Optimization, Evaluation

# In[ ]:


model_output = model(tf_facts,tf_questions)

# l2 regularization
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
regularization = tf.contrib.layers.apply_regularization(regularizer, reg_variables)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=tf_answers))+regularization

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

model_output = tf.nn.softmax(model_output)

#Evaluate model
correct_pred = tf.equal(tf.cast(tf.argmax(model_output,1),tf.int32),tf_answers)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = tf.argmax(model_output,1)

# Initializing the variables
init = tf.global_variables_initializer()


# ### Training....

# In[ ]:


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 

    sess.run(init) #initialize all variables
    step = 1   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_acc=0
    best_val_loss=2**30
    prev_val_acc=0
    patience = 20
    impatience = 0
    display_step = 20
    min_epoch = 50
            
    batch_size = 128
    
    while step <= epochs:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0

        batches_train_fact_stories,batches_train_questions,batches_train_answers = create_batches(train_fact_stories,train_questions,train_answers,batch_size)
        
        for i in xrange(len(batches_train_questions)):
            
            # Run optimization operation (backpropagation)
            _,loss,acc = sess.run([optimizer,cost,accuracy],
                                       feed_dict={tf_facts: batches_train_fact_stories[i], 
                                                  tf_questions: batches_train_questions[i], 
                                                  tf_answers: batches_train_answers[i],
                                                  keep_prob: 0.9})

            total_loss += loss
            total_acc += acc
                
            if i%display_step == 0:
                print "Iter "+str(i)+", Loss= "+                      "{:.3f}".format(loss)+", Accuracy= "+                      "{:.3f}".format(acc*100)
                        
        avg_loss = total_loss/len(batches_train_questions) 
        avg_acc = total_acc/len(batches_train_questions)  
        
        loss_list.append(avg_loss) 
        acc_list.append(avg_acc) 

        val_batch_size = 100 #(should be able to divide total no. of validation samples without remainder)
        batches_val_fact_stories,batches_val_questions,batches_val_answers = create_batches(val_fact_stories,val_questions,val_answers,val_batch_size)
        
        for i in xrange(len(batches_val_questions)):
            val_loss, val_acc = sess.run([cost, accuracy], 
                                         feed_dict={tf_facts: batches_val_fact_stories[i], 
                                                    tf_questions: batches_val_questions[i], 
                                                    tf_answers: batches_val_answers[i],
                                                    keep_prob: 1})
            total_val_loss += val_loss
            total_val_acc += val_acc
                      
            
        avg_val_loss = total_val_loss/len(batches_val_questions) 
        avg_val_acc = total_val_acc/len(batches_val_questions) 
             
        val_loss_list.append(avg_val_loss) 
        val_acc_list.append(avg_val_acc) 
    

        print "\nEpoch " + str(step) + ", Validation Loss= " +                 "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " +                 "{:.3f}%".format(avg_val_acc*100)+""
        print "Epoch " + str(step) + ", Average Training Loss= " +               "{:.3f}".format(avg_loss) + ", Average Training Accuracy= " +               "{:.3f}%".format(avg_acc*100)+""
        
        impatience += 1
        
        if avg_val_acc >= best_val_acc:
            best_val_acc = avg_val_acc
            saver.save(sess, 'DMN_Model_Backup/model.ckpt') 
            print "Checkpoint created!"
    
        if avg_val_loss <= best_val_loss: 
            impatience=0
            best_val_loss = avg_val_loss

        
        if impatience > patience and step>min_epoch:
            print "\nEarly Stopping since best validation loss not decreasing for "+str(patience)+" epochs."
            break
            
        print ""
        step += 1
        
    
        
    print "\nOptimization Finished!\n"
    
    print "Best Validation Accuracy: %.3f%%"%((best_val_acc*100))
    


# In[ ]:


#Saving logs about change of training and validation loss and accuracy over epochs in another file.

import h5py

file = h5py.File('Training_logs_customQA.h5','w')
file.create_dataset('val_acc', data=np.array(val_acc_list))
file.create_dataset('val_loss', data=np.array(val_loss_list))
file.create_dataset('acc', data=np.array(acc_list))
file.create_dataset('loss', data=np.array(loss_list))

file.close()


# In[ ]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

log = h5py.File('Training_logs_customQA.h5','r+') # Loading logs about change of training and validation loss and accuracy over epochs

y1 = log['val_acc'][...]
y2 = log['acc'][...]

x = np.arange(1,len(y1)+1,1) # (1 = starting epoch, len(y1) = no. of epochs, 1 = step) 

plt.plot(x,y1,'b',label='Validation Accuracy') 
plt.plot(x,y2,'r',label='Training Accuracy')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.show()

y1 = log['val_loss'][...]
y2 = log['loss'][...]

plt.plot(x,y1,'b',label='Validation Loss')
plt.plot(x,y2,'r',label='Training Loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.show()


# ### Testing...

# In[ ]:


with tf.Session() as sess: # Begin session
    
    print 'Loading pre-trained weights for the model...'
    saver = tf.train.Saver()
    saver.restore(sess, 'CustomQA_Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    print '\nRESTORATION COMPLETE\n'
    
    print 'Testing Model Performance...'
    
    total_test_loss = 0
    total_test_acc = 0
    
    test_batch_size = 100 #(should be able to divide total no. of test samples without remainder)
    batches_test_fact_stories,batches_test_questions,batches_test_answers = create_batches(test_fact_stories,test_questions,test_answers,test_batch_size)
        
    for i in xrange(len(batches_test_questions)):
        test_loss, test_acc = sess.run([cost, accuracy], 
                                        feed_dict={tf_facts: batches_test_fact_stories[i], 
                                                   tf_questions: batches_test_questions[i], 
                                                   tf_answers: batches_test_answers[i],
                                                   keep_prob: 1})
        total_test_loss += test_loss
        total_test_acc += test_acc
                      
            
    avg_test_loss = total_test_loss/len(batches_test_questions) 
    avg_test_acc = total_test_acc/len(batches_test_questions) 


    print "\nTest Loss= " +           "{:.3f}".format(avg_test_loss) + ", Test Accuracy= " +           "{:.3f}%".format(avg_test_acc*100)+""

