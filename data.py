import random
from word_utils import *
from number_utils import * 
from word_utils import nphrase_to_tensor

class NumbersDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.nphrase_to_numbers = {} # e.g "three hundred thirty two" 332

        with open(self.dataset_path, 'r') as f:
            for line in f: # Gives us the first line 
                lst = line.strip("\n").split(" ") #strips the new line and then the delimeter is the space
                # lst = ["21", 'twenty, 'one']

                number = int(lst[0])
                nphrase = " ".join(lst[1:]) # This joins all the other numbers into 1 string in the list
                self.nphrase_to_numbers[nphrase] = number

        self.nphrases = list(self.nphrase_to_numbers.keys()) # In a dictionary there are keys and values, the values are like 1,2,3 and keys are abc
        self.size = len(self.nphrases)

    def __len__(self):
        return self.size # this allows 

    def __getitem__(self, idx): # this allows us to treat our class with functionality so we can index into it ! very cool
        # get item using those words is a must its a python thing
        idx = random.randint(0,self.size-1)
        nphrase = self.nphrases[idx]
        nphrase_tensor = nphrase_to_tensor(nphrase)
        number = self.nphrase_to_numbers[nphrase]
        number_indicies = number_to_indices(number)
        number_indicies = torch.LongTensor([number_indicies])

        return nphrase, number, nphrase_tensor, number_indicies

    def get_random_sample(self):
        idx = random.randint(0,self.size-1)
        return self[idx] # self refers to the class index. by doing idx, this calls __get__item
        # whenever you put brackets, it calls get_item method

        




