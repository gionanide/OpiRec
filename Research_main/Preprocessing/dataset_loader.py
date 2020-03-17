#!usr/bin/python
import torch
from torch.utils.data import Dataset
import pickle

'''
Make this class in order to read the sampes from the files override the Pytorch Dataset class
'''

class Dataset_yelp(Dataset):

        def __init__(self, file_path, transform=None):
                ''' Initialization '''
                #read the dataset from the file
                self.dataset = pickle.load(open(file_path,'rb'))
                #assign values to the fields
                self.user_reviews = self.dataset.user_reviews
                self.product_reviews = self.dataset.product_reviews
                self.neighbourhood_reviews = self.dataset.neighbourhood_reviews
                self.product_ratings = self.dataset.product_ratings
                self.text_reviews = self.dataset.text_reviews
                self.rating_reviews = self.dataset.rating_reviews
                #define the transformation
                self.transform = transform
                
        def __len__(self):
                ''' Get the length of the dataset '''
                #print(len(self.rating_reviews))
                return len(self.rating_reviews)
                
        def __getitem__(self, index):
                ''' Generates one sample of data '''
                #assign all the fields
                user_reviews = torch.tensor(self.user_reviews[index], dtype=torch.float32)
                product_reviews = torch.tensor(self.product_reviews[index], dtype=torch.float32)
                neighbourhood_reviews = torch.tensor(self.neighbourhood_reviews[index], dtype=torch.float32)
                product_ratings = np.array(self.product_ratings[index])
                text_reviews = self.text_reviews[index][0]
                rating_reviews = self.rating_reviews[index]
                decoder_inputs = self.text_reviews[index][0]
                
                return user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs
                
                
                
                
                
                
