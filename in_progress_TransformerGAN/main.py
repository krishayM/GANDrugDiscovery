#Edited from https://github.com/suragnair/seqGAN

from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

from generator_attention import Generator_attention as ga
import discriminator
import oracle
import helpers
from jak_helpers import *

from tqdm import tqdm



CUDA = False

REAL_DATA_PATH = '../jak_data/JAK2New.txt'
ORACLE_DATA_PATH = '../jak_data/jak.pt'

#These must be initialized
VOCAB_SIZE = None
MAX_SEQ_LEN = None
START_LETTER = None

BATCH_SIZE = 5 #Transformer is slow. Used to be 32.

#Commented epochs for debugging
MLE_TRAIN_EPOCHS = 1 #10 #100
DISC_STEPS = 1 #50
DISC_EPOCHS = 1 #3
ADV_DISC_STEPS = 1 #5
ADV_DISC_EPOCHS = 1 #3

ADV_TRAIN_EPOCHS = 50

'''
Reduced to speed debugging.
'''
POS_NEG_SAMPLES = 1 #15 #1000 #1000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

#oracle_samples_path = './oracle_samples.trc'
oracle_samples_path = '../jak_data/jak.pt'

'''
Ultimately, we don't want a state dict for the oracle.
'''
#oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'

pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

#Called as:
#train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)
def train_generator_MLE(gen, gen_opt, oracle, epochs):
    """
    Max Likelihood Pretraining for the generator
    gen: generator (starts as random, trains to get gud.)
    gen_opt: generator optimizer
    oracle: Generates real data. Ideally, a perfect agent - in practice.
    real_data_samples: samples taken from the oracle.
         - Oracle can now generate this.
    epochs: times to iterate through the dataset.
    """
    
    real_data_samples = oracle.data
    
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        '''
        Transformer is slow, so we're going to reduce POS_NEG_SAMPLES by a LOT.
        '''
        #tgdm is used for progress bars
        for i in tqdm(range(0, POS_NEG_SAMPLES, BATCH_SIZE)):
            '''
            Train the generator to understand how a given character follows those preceding it.
            Do this by taking ground truth data and shifting it against itself by 1, 
            effectively training the Generator to resemble the Oracle in its output.
            '''
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER, gpu=CUDA)
            gen_opt.zero_grad()#zero_grad is used to set gradients of all parameters to 0 so they only accumulate while computing loss in backprop
            
            '''
            When we use Transformer, update this line to run loss on new model.
            '''
            loss = gen.batchNLLLoss(inp, target)
            
            
            loss.backward()#calculates the backward gradient across every node
            gen_opt.step()#performs a parameter update based on the gradient of the loss

            total_loss += loss.data.item()

            #Print progress
            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        
        '''
        Likely not needed, as Oracle is no longer a generator.
        '''
        # sample from generator and compute oracle NLL
        #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                           start_letter=START_LETTER, gpu=CUDA)

        #print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        '''
        Redefined progress print omitting Oracle Loss:
        '''
        print(f'average_train_NLL = {round(total_loss,4)}')

def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()
        
        print(f'PG Gen Loss: {pg_loss}')

    '''
    We've done away with Oracle Loss, so... DIE!
    '''
    # sample from generator and compute oracle NLL
    #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                               start_letter=START_LETTER, gpu=CUDA)

    #print(' oracle_sample_NLL = %.4f' % oracle_loss)

#Function called as: train_discriminator(dis, dis_optimizer, gen, oracle, 50, 3)
def train_discriminator(discriminator, dis_opt, generator, oracle_obj, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    
    real_data_samples = oracle_obj.data

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle_obj.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        r = helpers.batchwise_sample(oracle_obj, POS_NEG_SAMPLES, BATCH_SIZE)
        
        #get discriminator inputs (batch of sequences) and targets (Ys - batch of binary values)
        
        dis_inp, dis_target = helpers.prepare_discriminator_data(r, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                
                out = discriminator.batchClassify(inp)
                
                #print(inp.shape, out.shape)
                #print(target)
                #print(inp, out)
                
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':
    
    #Encode the real data to int tokens, and set the key global variables from that encoding.
    VOCAB_SIZE, MAX_SEQ_LEN, START_LETTER = encode_data(REAL_DATA_PATH, ORACLE_DATA_PATH)
    
    '''
    TODO: Oracle is currently a seeded random generator. We need to turn it into something that 
    generates real data in a token-wise manner.
    '''
    #oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    oracle_obj = oracle.Oracle(VOCAB_SIZE, MAX_SEQ_LEN, oracle_samples_path)
    
    '''
    This is the 'seed' part of the oracle. We likely won't need it.
    '''
    #oracle.load_state_dict(torch.load(oracle_state_dict_path))
    
    '''
    Oracle samples should be the real data - i.e., the JAK data.
    '''
    #oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    '''
    Look into using the following:
    '''
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    '''
    Gen and Dis are both randomly initialized.
    '''
    '''
    Changed gen from Generator to Transformer:
    '''
    #gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    gen = ga(VOCAB_SIZE, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, MAX_SEQ_LEN, BATCH_SIZE, True, ORACLE_DATA_PATH)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle_obj = oracle_obj.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        #oracle_samples = oracle_samples.cuda()
        

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    
    #Modification to train on JAK data.
    #From : https://github.com/suragnair/seqGAN/issues/10
    #Func defined above!
    train_generator_MLE(gen, gen_optimizer, oracle_obj, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    
    #Func defined above!
    
    
    train_discriminator(dis, dis_optimizer, gen, oracle_obj, DISC_STEPS, DISC_EPOCHS)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))
    filenameGEN = 'finalized_gen.sav'
    pickle.dump(gen, open(filenameGEN, 'wb'))
    
    filenameDIS = 'finalized_discrim.sav'
    pickle.dump(dis, open(filenameDIS, 'wb'))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    '''
    Don't need oracle loss anymore.
    '''
    #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle_obj, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                           start_letter=START_LETTER, gpu=CUDA)
    #print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle_obj, dis, 1)
        
        sample_gen_output = gen.sample(1)
        sample_real_output = oracle_obj.sample(1)
        print(f'real: {sample_real_output}')
        print(f'fake: {sample_gen_output}')

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, gen, oracle_obj, ADV_DISC_STEPS, ADV_DISC_EPOCHS)
        
    filenameGEN2 = 'finalized_genAfter.sav'
    pickle.dump(gen, open(filenameGEN2, 'wb'))
    
    filenameDIS2 = 'finalized_discrimAfter.sav'
    pickle.dump(dis, open(filenameDIS2, 'wb'))
        