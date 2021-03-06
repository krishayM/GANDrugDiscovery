# -*- coding: utf-8 -*-

import os
import random
import sys
# sys.path.insert(0, './transformer')
import transformer
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.data_iter import GenDataIter

from tqdm import tqdm

class Generator_attention(nn.Module):
    """Generator """
    
    #Called as:        (VOCAB_SIZE, g_emb_dim, g_hidden_dim, g_sequence_len, BATCH_SIZE, opt.cuda, POSITIVE_FILE)
    def __init__(self, num_emb, emb_dim, hidden_dim, seq_len, batch_size, use_cuda, real_data_path, test_mode = False):
        super(Generator_attention, self).__init__()
        # Constants Initialization
        self.SOS_Index = 0
        self.EOS_Index = 1
        self.PAD_Index = 2
        self.real_data_path = real_data_path
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        
        '''self.use_cuda = use_cuda'''
        
        # Embeddings
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.model = transformer.Transformer(
            self.emb,
            self.PAD_Index,
            self.emb.num_embeddings,
            max_seq_len = max(seq_len, seq_len)
        )
        self.test_mode = test_mode
        if not test_mode:
            self.data_loader = GenDataIter(self.real_data_path, batch_size)
            self.data_loader.reset()
            
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        return self.model(x, x)
       

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)))
        return pred, h, c


    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        '''if self.use_cuda:
            h, c = h.cuda(), c.cuda()'''
        return h, c
    
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, sample_size, x=torch.tensor([])):
        if self.test_mode:
            print('In Test mode')
            return None
            
        if self.data_loader.idx >= self.data_loader.data_num:
            self.data_loader.reset()
        if len(x.shape) > 1:
            input_seq = x
        else:
            input_seq = self.data_loader.next()[0][:min(sample_size, self.batch_size)]
        input_seq = input_seq#.cuda()
        sampled_output = transformer.sample_output(self.model, input_seq, self.EOS_Index, self.PAD_Index, input_seq.shape[1])
        return sampled_output
        
    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        
        '''
        Is this permutation needed?
        Yes it is.
        '''
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        
        h = self.init_hidden(batch_size)
        

        loss = 0
        '''
        Transformer model maps a sequence to a sequence, not an int to the following int.
        We need to change how the data is fed in. 
        Okay. Given that our input and target are just the same sequence (sampled randomly from real data),
        we can call forward on the whole sequence, and pass out and target to the function. No need for a for loop.
        '''
        out = self.forward(inp)
        
        '''
        The output of self.forward() is a tensor of shape [MAX_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE] 
        i.e. [Seq_Axis, Batch_Axis, Classes_Axis]
        No need to take the argmax! NLLoss expected softmaxed class vectors.
        '''
        '''
        We can grab the output first, then calc NLLLoss on each element in the output.
        '''
        
        #print(out.shape)
        #print(target.shape)
        
        '''
        target shape is [Seq_Len x Batch]
        '''
        
        
        for i in range(seq_len):
            p = out[i] #Shape (Batch x Classes)
            t = target[i]
            '''
            NLLoss takes:
            inp: Shape (Batch x Classes)
            target: Shape (Batch)
            '''
            
            loss += loss_fn(p, t)

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)
    
        out = self.forward(inp)
        
        loss = 0
        for i in range(seq_len):
            p = out[i]
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -p[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size