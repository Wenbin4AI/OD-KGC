#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples) 
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self): # Return the number of samples
        return self.len
    
    def __getitem__(self, idx):# Return dataset and labels
        positive_sample = self.triples[idx] 
        # Create negative samples for this positive sample
        head, relation, tail = positive_sample
        # subsampling_weight: subsampling weight
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size: # self.negative_sample_size: 256
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2) # Randomly generate 256*2 entity indices. Why 256*2? Because we need to remove correct triples later and ensure we have 256 left.
            # np.random.randint(low, high, size) returns a list of random integers between low and high (exclusive) of the specified size. Here, low is the number of entities.
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch': # Treat the randomly generated numbers as tail entity indices
                # np.in1d returns a boolean array of the same length as a that is True where an element of a is in b and False otherwise. Here, mask is a boolean array of length 512.
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)], # Find the indices of the positive samples in the randomly generated negative samples and return a boolean array.
                    # This way, we find the indices of the correct triples in the randomly generated negative samples. The mask is used to remove these correct samples.
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask] # After generating the mask, remove the correct samples.
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size] # Concatenate the negative samples

        negative_sample = torch.LongTensor(negative_sample) # Convert to LongTensor. Indexing vectors must be LongTensors. | negative_sample: tensor: (256,) These are the 256 incorrect tail/head entities for this positive triple.

        positive_sample = torch.LongTensor(positive_sample) # Convert to LongTensor | positive_sample: tensor: (3,) This is the correct triple.

        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data): # data={list:1024}  Each element is a tuple:4, where each element of the tuple is the correct triple, the incorrect triples' head/tail entities, the subsampling weight, and the mode
        positive_sample = torch.stack([_[0] for _ in data], dim=0) # tensor: (1024,3) The correct triples
        negative_sample = torch.stack([_[1] for _ in data], dim=0) # tensor: (1024,256) The incorrect triples' tail entities
        subsample_weight = torch.cat([_[2] for _ in data], dim=0) # tensor: (1024,)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4): # Corresponds to self.count    Why is start 4?
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples): # Count what tails (head entity, relation) has and what heads (relation, tail entity) has, and return dictionaries
        '''
        Build a dictionary of true triples that will be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail) # {head, relation: [list of tails]}
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head) # {relation, tail: [list of heads]}
        # Convert lists to ndarray format {(123, 456): [112, 121, 111]}  >>>  {(123, 456): array([112, 121, 111])}
        for relation, tail in true_head: # np.in1d is used later, so convert to ndarray
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)]))) # Remove duplicates and use np.array(list) to create an ndarray
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples) # All true triples without duplicates
        self.triples = triples # Test triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode # head-batch/tail-batch

    def __len__(self):
        return self.len
    '''
    There are no negative samples in TestDataset, but we need to select and rank from all head entities
    [exp1 if condition else exp2 for x in data]
    The if...else here mainly serves to assign values. When the data in data satisfies the if condition, it is processed as exp1, otherwise it is processed as exp2, and finally a data list is generated
    Iterate through all entities, if not in all true triples, return an element (0,rand_head), if in triples, return an element (-1, head), head is a fixed value, rand_head is the element being iterated
    When head = 11267
    We get something like tmp = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(-1,11267),(0,7)........(0,215),(-1,11267),......(0,11266),(-1,11267)(0,11268).......(0,14951)]
    After tmp[head] = (0, head) 
    We finally get something like tmp = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(-1,11267),(0,7)........(0,215),(-1,11267),......(0,11266),(0,11267)(0,11268).......(0,14951)]
    That is, the element at position 11267 becomes (0,rand_head) in normal sequential order
    '''
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        # The if...else here mainly serves to assign values. 
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head) 
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)  # tensor: (14951,2)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1] # tensor(14951,)

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader): 
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
