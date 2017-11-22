
# coding: utf-8

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



# In[2]:


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

word = 'frog'

most_similars = most_similar_eucli(word2vec(word))

print "TOP TEN MOST SIMILAR WORDS TO '"+str(word)+"':\n"
for i in xrange(0,10):
    print str(i+1)+". "+str(vocab[most_similars[i]])

def vec2word(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli(np.asarray(vec,np.float32))
    return vocab[most_similars[0]]


# In[3]:


import string
# Data related to basic induction training and testing from QA bAbi tasks dataset will be used.
# (https://research.fb.com/downloads/babi/)

filename = 'qa16_basic-induction_train.txt' 

fact_story = [] 
question = []   
answer = []   

    
#max_fact_len = 6
#max_question_len = 5


def extract_info(filename):  
        
    fact_story = [] 
    fact_stories = []
    questions = []
    answers = []
    #PAD = word2vec('<PAD>')

    file = open(filename,'r')
    for line in file.readlines(): 
        
        flag_end_story = 0 
        line = line.lower() 
        if '?' in line:
            #q for question, a for answer.
            flag_end_story=1
            qa = line.strip().split('\t')
            q = qa[0]
            a = qa[1]
            q = q.translate(None, string.punctuation)
            a = a.translate(None, string.punctuation)
            q = q.strip().split(' ')
            a = a.strip().split(' ')
            q = q[1:]
            q = map(word2vec,q)
            questions.append(q)
            answers.append(map(vocab.index,a))
            
        else: 
            line = line.translate(None, string.punctuation)
            fact = line.strip().split(' ') 
            fact = fact[1:]
            fact = map(word2vec,fact)
            #for i in xrange(len(fact),max_fact_len):
                #fact.append(PAD)
            fact_story.append(fact)

        if flag_end_story == 1: 
            fact_stories.append(fact_story)  
            fact_story = [] 
            
    file.close()
        
    return fact_stories,questions,answers

fact_stories,questions,answers = extract_info(filename)

filename = 'qa16_basic-induction_test.txt' 

test_fact_stories,test_questions,test_answers = extract_info(filename)


# In[5]:


max_fact_len = 6
max_question_len = 5

PAD = word2vec('<PAD>')

for i in xrange(0,len(questions)):
    questions_len = len(questions[i])
    for j in xrange(questions_len,max_question_len):
        questions[i].append(PAD)
    for j in xrange(0,len(fact_stories[i])):
        fact_len = len(fact_stories[i][j])
        for k in xrange(fact_len,max_fact_len):
            fact_stories[i][j].append(PAD)
 


# In[6]:


for i in xrange(0,len(test_questions)):
    questions_len = len(test_questions[i])
    for j in xrange(questions_len,max_question_len):
        test_questions[i].append(PAD)
    for j in xrange(0,len(test_fact_stories[i])):
        fact_len = len(test_fact_stories[i][j])
        for k in xrange(fact_len,max_fact_len):
            test_fact_stories[i][j].append(PAD)


# In[7]:


fact_stories = np.asarray(fact_stories,np.float32)
print fact_stories.shape
questions = np.asarray(questions,np.float32)
print questions.shape
answers = np.asarray(answers,np.float32)
print answers.shape
test_fact_stories = np.asarray(test_fact_stories,np.float32)
print test_fact_stories.shape
test_questions = np.asarray(test_questions,np.float32)
print test_questions.shape
test_answers = np.asarray(test_answers,np.float32)
print test_answers.shape


# In[8]:


#Saving processed data in another file.

import pickle

PICK = [fact_stories,questions,answers,test_fact_stories,test_questions,test_answers]

with open('embeddingPICKLE', 'wb') as fp:
    pickle.dump(PICK, fp)

