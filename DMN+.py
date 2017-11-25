
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

with open ('embeddingPICKLE', 'rb') as fp:
    processed_data = pickle.load(fp)

fact_stories = processed_data[0]
questions = processed_data[1]
answers = np.reshape(processed_data[2],(len(processed_data[2])))
test_fact_stories = processed_data[3]
test_questions = processed_data[4]
test_answers = np.reshape(processed_data[5],(len(processed_data[5])))

print fact_stories.shape
print questions.shape
print answers.shape
print test_fact_stories.shape
print test_questions.shape
print test_answers.shape
    


# In[2]:


print map(vec2word,fact_stories[0][0])


# In[3]:


print map(vec2word,questions[0])


# ### CREATING TRAINING AND VALIDATION DATA

# In[4]:


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


# ### SENTENCE READING LAYER IMPLEMENTED BEFOREHAND 
# 
# Positionally encode the word vectors in each sentence, and combine all the words in the sentence to create a fixed sized vector representation for the sentence.
# 
# "sentence embedding"

# In[5]:


def sentence_reader(fact_stories): #positional_encoder
    
    pe_fact_stories = np.zeros((fact_stories.shape[0],fact_stories.shape[1],word_vec_dim),np.float32)
    
    for fact_story_index in xrange(0,len(fact_stories)):
        for fact_index in xrange(0,len(fact_stories[fact_story_index])):
            
            M = len(fact_stories[fact_story_index,fact_index]) #length of sentence (fact)
            l = np.zeros((word_vec_dim),np.float32) 
            
            # ljd = (1 − j/M) − (d/D)(1 − 2j/M),
            
            for word_position in xrange(0,M):
                for dimension in xrange(0,word_vec_dim):
                    
                    j = word_position + 1 # making position start from 1 instead of 0
                    d = dimension + 1 # making dimensions start from 1 isntead of 0 (1-50 instead of 0-49)
                    
                    l[dimension] = (1-(j/M)) - (d/word_vec_dim)*(1-2*(j/M))
                
                fact_stories[fact_story_index,fact_index,word_position] = np.multiply(l,fact_stories[fact_story_index,fact_index,word_position])

            pe_fact_stories[fact_story_index,fact_index] = np.sum(fact_stories[fact_story_index,fact_index],0)

    return pe_fact_stories

train_fact_stories = sentence_reader(train_fact_stories)
val_fact_stories = sentence_reader(val_fact_stories)
test_fact_stories = sentence_reader(test_fact_stories)
                
        


# In[6]:


print train_fact_stories.shape
print val_fact_stories.shape
print test_fact_stories.shape


# ### Function to create randomized batches

# In[7]:


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
        batch_fact_stories = np.transpose(batch_fact_stories,[1,0,2])
        #result = number of facts x batch_size x fact sentence size x word vector size
        
        batch_questions = np.asarray(batch_questions,np.float32)
        batch_questions = np.transpose(batch_questions,[1,0,2])
        #result = question_length x batch_size x fact sentence size x word vector size
        
        batches_fact_stories.append(batch_fact_stories)
        batches_questions.append(batch_questions)
        batches_answers.append(batch_answers)
        
        i+=batch_size
        
    batches_fact_stories = np.asarray(batches_fact_stories,np.float32)
    batches_questions = np.asarray(batches_questions,np.float32)
    batches_answers = np.asarray(batches_answers,np.float32)
    
    return batches_fact_stories,batches_questions,batches_answers
    


# ### Hyperparameters

# In[8]:


import tensorflow as tf

# Tensorflow placeholders

tf_facts = tf.placeholder(tf.float32, [None,None,word_vec_dim])
tf_questions = tf.placeholder(tf.float32, [None,None,word_vec_dim])
tf_answers = tf.placeholder(tf.int32,[None])
keep_prob = tf.placeholder(tf.float32)

#hyperparameters
epochs = 256
learning_rate = 0.001
hidden_size = 100
passes = 3
beta = 0.005 #l2 regularization scale


# ### Low level api implementation of GRU
# 
# Returns a tensor of all the hidden states

# In[9]:


def GRU(inp,hidden,
        wz,uz,bz,
        wr,ur,br,
        w,u,b,
        seq_len):

    hidden_lists = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    i=0
    
    def cond(i,hidden,hidden_lists):
        return i < seq_len
    
    def body(i,hidden,hidden_lists):
        
        x = inp[i]

        # GRU EQUATIONS:
        z = tf.sigmoid( tf.matmul(x,wz) + tf.matmul(hidden,uz) + bz)
        r = tf.sigmoid( tf.matmul(x,wr) + tf.matmul(hidden,ur) + br)
        h_ = tf.tanh( tf.matmul(x,w) + tf.multiply(r,tf.matmul(hidden,u)) + b)
        hidden = tf.multiply(z,hidden) + tf.multiply((1-z),h_)

        hidden_lists = hidden_lists.write(i,hidden)
        
        return i+1,hidden,hidden_lists
    
    _,_,hidden_lists = tf.while_loop(cond,body,[i,hidden,hidden_lists])
    
    return hidden_lists.stack()
        


# ### Attention based GRU as used in DMN+ model
# 
# Returns only the final hidden state.

# In[10]:


def attention_based_GRU(inp,hidden,
                        wr,ur,br,
                        w,u,b,
                        g,seq_len):
    
    i=0
    
    def cond(i,hidden):
        return i < seq_len
    
    def body(i,hidden):
        
        x = inp[i]

        # GRU EQUATIONS:
        r = tf.sigmoid( tf.matmul(x,wr) + tf.matmul(hidden,ur) + br)
        h_ = tf.tanh( tf.matmul(x,w) + tf.multiply(r,tf.matmul(hidden,u)) + b)
        hidden = tf.multiply(g[i],hidden) + tf.multiply((1-g[i]),h_)
        
        return i+1,hidden
    
    _,hidden = tf.while_loop(cond,body,[i,hidden])
    
    return hidden
        


# ### All the trainable parameters initialized here

# In[11]:



# Parameters

# FORWARD GRU PARAMETERS FOR INPUT MODULE

value = np.zeros((hidden_size),np.float32)
init = tf.constant_initializer(value)
regularizer = tf.contrib.layers.l2_regularizer(scale=beta)

wzf = tf.get_variable("wzf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer= regularizer)
uzf = tf.get_variable("uzf", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
bzf = tf.get_variable("bzf", shape=[hidden_size],initializer=init)

wrf = tf.get_variable("wrf", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
urf = tf.get_variable("urf", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
brf = tf.get_variable("brf", shape=[hidden_size],initializer=init)

wf = tf.get_variable("wf", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
uf = tf.get_variable("uf", shape=[hidden_size, hidden_size],
                     initializer=tf.orthogonal_initializer(),
                     regularizer=regularizer)
bf = tf.get_variable("bf", shape=[hidden_size],initializer=init)

# BACKWARD GRU PARAMETERS FOR INPUT MODULE

wzb = tf.get_variable("wzb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
uzb = tf.get_variable("uzb", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
bzb = tf.get_variable("bzb", shape=[hidden_size],
                      initializer=init)

wrb = tf.get_variable("wrb", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
urb = tf.get_variable("urb", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
brb = tf.get_variable("brb", shape=[hidden_size],initializer=init)

wb = tf.get_variable("wb", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
ub = tf.get_variable("ub", shape=[hidden_size, hidden_size],
                     initializer=tf.orthogonal_initializer(),
                     regularizer=regularizer)
bb = tf.get_variable("bb", shape=[hidden_size],initializer=init)

# GRU PARAMETERS FOR QUESTION MODULE (TO ENCODE THE QUESTIONS)

wzq = tf.get_variable("wzq", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
uzq = tf.get_variable("uzq", shape=[hidden_size, hidden_size],initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
bzq = tf.get_variable("bzq", shape=[hidden_size],initializer=init)

wrq = tf.get_variable("wrq", shape=[word_vec_dim, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
urq = tf.get_variable("urq", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
brq = tf.get_variable("brq", shape=[hidden_size],initializer=init)

wq = tf.get_variable("wq", shape=[word_vec_dim, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
uq = tf.get_variable("uq", shape=[hidden_size, hidden_size],
                     initializer=tf.orthogonal_initializer(),
                     regularizer=regularizer)
bq = tf.get_variable("bq", shape=[hidden_size],initializer=init)


# EPISODIC MEMORY

inter_neurons = 1024
w1 = tf.get_variable("w1", shape=[hidden_size*4, inter_neurons],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
b1 = tf.get_variable("b1", shape=[inter_neurons],initializer=tf.constant_initializer(0))
w2 = tf.get_variable("w2", shape=[inter_neurons,1],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
b2 = tf.get_variable("b2", shape=[1],initializer=tf.constant_initializer(0))

# ATTENTION BASED GRU PARAMETERS

wratt = tf.get_variable("wratt", shape=[hidden_size,hidden_size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=regularizer)
uratt = tf.get_variable("uratt", shape=[hidden_size,hidden_size],
                        initializer=tf.orthogonal_initializer(),
                        regularizer=regularizer)
bratt = tf.get_variable("bratt", shape=[hidden_size],initializer=init)

watt = tf.get_variable("watt", shape=[hidden_size,hidden_size],
                       initializer=tf.contrib.layers.xavier_initializer(),
                       regularizer=regularizer)
uatt = tf.get_variable("uatt", shape=[hidden_size, hidden_size],
                       initializer=tf.orthogonal_initializer(),
                       regularizer=regularizer)
batt = tf.get_variable("batt", shape=[hidden_size],initializer=init)

# MEMORY UPDATE PARAMETERS

#wt = tf.get_variable("wt1", shape=[passes,3],initializer=tf.random_uniform_initializer())
wt = tf.get_variable("wt", shape=[passes,hidden_size*3,hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
bt = tf.get_variable("bt", shape=[passes,hidden_size],initializer=tf.constant_initializer(np.zeros((passes,hidden_size),np.float32)))

# Answer module

# GRU PARAMETERS FOR ANSWER MODULE

wza = tf.get_variable("wza", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
uza = tf.get_variable("uza", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
bza = tf.get_variable("bza", shape=[hidden_size],initializer=init)

wra = tf.get_variable("wra", shape=[hidden_size, hidden_size],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)
ura = tf.get_variable("ura", shape=[hidden_size, hidden_size],
                      initializer=tf.orthogonal_initializer(),
                      regularizer=regularizer)
bra = tf.get_variable("bra", shape=[hidden_size],initializer=init)

wa = tf.get_variable("wa", shape=[hidden_size, hidden_size],
                     initializer=tf.contrib.layers.xavier_initializer(),
                     regularizer=regularizer)
ua = tf.get_variable("ua", shape=[hidden_size, hidden_size],
                     initializer=tf.orthogonal_initializer(),
                     regularizer=regularizer)
ba = tf.get_variable("ba", shape=[hidden_size],initializer=init)

# Parameter to convert output as of now to a probability distribution. 
    
wa1 = tf.get_variable("wa1", shape=[hidden_size,len(vocab)],
                      initializer=tf.contrib.layers.xavier_initializer(),
                      regularizer=regularizer)    


# ### Dynamic Memory Network+ Model Definition

# In[12]:


def DMN_plus(tf_facts,tf_questions):
    
    facts_num = tf.shape(tf_facts)[0]
    tf_batch_size = tf.shape(tf_questions)[1]
    question_len = tf.shape(tf_questions)[0]
    
    hidden = tf.zeros([tf_batch_size,hidden_size],tf.float32)

    
    tf_facts = tf.nn.dropout(tf_facts,keep_prob)
    
    # Input Module
    # input fusion layer 
    # bidirectional GRU
    
    forward = GRU(tf_facts,hidden,
                  wzf,uzf,bzf,
                  wrf,urf,brf,
                  wf,uf,bf,
                  facts_num)
    
    backward = GRU(tf.reverse(tf_facts,[0]),hidden,
                   wzf,uzf,bzf,
                   wrf,urf,brf,
                   wf,uf,bf,
                   facts_num)
    
    encoded_input = forward + backward

    # Question Module
    
    question_representation = GRU(tf_questions,hidden,
                                  wzq,uzq,bzq,
                                  wrq,urq,brq,
                                  wq,uq,bq,
                                  question_len)
    
    #question_representation's current shape = question len x batch size x hidden size
    
    question_representation = question_representation[question_len-1]
    
    #^we will only use the final hidden state. 

    question_representation = tf.reshape(question_representation,[tf_batch_size,1,hidden_size])
    
    
    # Episodic Memory Module
    
    episodic_memory = question_representation
    
    encoded_input = tf.transpose(encoded_input,[1,0,2])
    #now shape = batch_size x facts_num x hidden_size
    
    
    i=0

    def cond(i,episodic_memory):
        return i < passes
    
    def body(i,episodic_memory):
        
        # Attention Mechanism
        
        Z1 = tf.multiply(encoded_input,question_representation)
        Z2 = tf.multiply(encoded_input,episodic_memory)
        Z3 = tf.abs(tf.subtract(encoded_input,question_representation))
        Z4 = tf.abs(tf.subtract(encoded_input,episodic_memory))
        
        Z = tf.concat([Z1,Z2,Z3,Z4],2)
        
        Z = tf.reshape(Z,[-1,4*hidden_size])
        Z = tf.add( tf.matmul( tf.tanh( tf.add( tf.matmul(Z,w1),b1 ) ),w2 ) , b2)
        Z = tf.reshape(Z,[tf_batch_size,facts_num])
        
        g = tf.nn.softmax(Z)
        g = tf.reshape(g,[tf_batch_size,facts_num])
        g = tf.transpose(g,[1,0])
        g = tf.reshape(g,[facts_num,tf_batch_size,1])
        
        context_vector = attention_based_GRU(tf.transpose(encoded_input,[1,0,2]),
                                             tf.reshape(episodic_memory,[tf_batch_size,hidden_size]),
                                             wratt,uratt,bratt,
                                             watt,uatt,batt,
                                             g,facts_num)
        
        context_vector = tf.reshape(context_vector,[tf_batch_size,1,hidden_size])
        
        # Episodic Memory Update
        
        concated = tf.concat([episodic_memory,context_vector,question_representation],2)
        concated = tf.reshape(concated,[-1,3*hidden_size])
        
        episodic_memory = tf.nn.relu(tf.matmul(concated,wt[i]) + bt[i])
        episodic_memory = tf.reshape(episodic_memory,[tf_batch_size,1,hidden_size])
        
        return i+1,episodic_memory
    
    
    _,episodic_memory = tf.while_loop(cond,body,[i,episodic_memory]) 
    
    # Answer module
    
    episodic_memory = tf.reshape(episodic_memory,[tf_batch_size,hidden_size])
    episodic_memory = tf.nn.dropout(episodic_memory,keep_prob)
    
    # sending in only the question as input. 
    # Only focusing on single word prediction, so no need of taking previous y into context. 
    # (because there will never be a previous y. No need to create <SOS> either.)

    question_representation = tf.transpose(question_representation,[1,0,2])
    question_representation = tf.nn.dropout(question_representation,keep_prob)
   
    y_state = GRU(question_representation,episodic_memory,
                  wza,uza,bza,
                  wra,ura,bra,
                  wa,ua,ba,1)
    
    y_state = y_state[0]
    y_state = tf.reshape(y_state,[tf_batch_size,hidden_size])
    y = tf.matmul(y_state,wa1) 
    
    return y


# ### Cost function, Evaluation, Optimization function 

# In[13]:


model_output = DMN_plus(tf_facts,tf_questions)


# l2 regularization
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
regularization = tf.contrib.layers.apply_regularization(regularizer, reg_variables)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=tf_answers))+regularization

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,centered=True).minimize(cost)

#Evaluate model
correct_pred = tf.equal(tf.cast(tf.argmax(model_output,1),tf.int32),tf_answers)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = tf.argmax(model_output,1)

# Initializing the variables
init = tf.global_variables_initializer()


# ### Training....

# In[14]:


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 1   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_loss=2**30
    prev_val_acc=0
    patience = 99 #a bit too much patience here....
    impatience = 0
    display_step = 20
            
    batch_size = 128
    
    while step <= epochs:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0

        batches_train_fact_stories,batches_train_questions,batches_train_answers = create_batches(train_fact_stories,train_questions,train_answers,batch_size)
        
        for i in xrange(len(batches_train_questions)):
            
            # Run optimization operation (backpropagation)
            _,loss,acc,pred = sess.run([optimizer,cost,accuracy,prediction],
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
            
        if avg_val_loss <= best_val_loss: # When better accuracy is received than previous best validation accuracy
            impatience = 0
            best_val_loss = avg_val_loss # update value of best validation accuracy received yet.
            saver.save(sess, 'DMN_Model_Backup/model.ckpt') # save_model including model variables (weights, biases etc.)
            print "Checkpoint created!"  
        
        if impatience > patience:
            print "Early Stopping since best validation loss not decreasing for "+str(patience)+" epochs."
            break
            
        print ""
        
        step += 1
        
    
        
    print "\nOptimization Finished!\n"
    
    print "Best Validation Loss: %.3f%%"%((best_val_loss))
    
    #The model can be run on test data set after this.
    #val_loss_list, val_acc_list, loss_list and acc_list can be used for plotting. 
    


# In[15]:


#Saving logs about change of training and validation loss and accuracy over epochs in another file.

import h5py

file = h5py.File('Training_logs_DMN_plus.h5','w')
file.create_dataset('val_acc', data=np.array(val_acc_list))
file.create_dataset('val_loss', data=np.array(val_loss_list))
file.create_dataset('acc', data=np.array(acc_list))
file.create_dataset('loss', data=np.array(loss_list))

file.close()


# In[16]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

log = h5py.File('Training_logs_DMN_plus.h5','r+') # Loading logs about change of training and validation loss and accuracy over epochs

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


# In[17]:


with tf.Session() as sess: # Begin session
    
    print 'Loading pre-trained weights for the model...'
    saver = tf.train.Saver()
    saver.restore(sess, 'DMN_Model_Backup/model.ckpt')
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
