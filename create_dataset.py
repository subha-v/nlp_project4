import random
from word_utils import *

MAX_NUMBER = 999
TRAIN_PROB = 0.8
VAL_PROB = 0.1
TEST_PROB = 0.1

#80% is for training 10% is for validation and 10% is for testing

def number_to_names():

    train_names, val_names, test_names = [],[],[]
    train_numbers, val_numbers, test_numbers = [],[],[]


    size = MAX_NUMBER+1
    numbers = list(range(0,MAX_NUMBER+1))

    for number in numbers:
        name = None
        if(number<20):
            name = vocab_map[number]
        elif(number<100):
            ones_number = number % 10
            tens_number = (number//10) * 10  # Integer division by 10
            if(ones_number>0):
                name = f"{vocab_map[tens_number]} {vocab_map[ones_number]}"
            else:
                name = vocab_map[tens_number]
        elif(number < 1000):
            hundreds_number = number // 100
            tens_number = ((number % 100) //10)
            ones_number = number % 10
            last_two = number % 100
            if(last_two> 10 and last_two <20):
                name = f"{vocab_map[hundreds_number*100]} {vocab_map[last_two]}"
            elif(tens_number == 0):
                if (ones_number == 0):
                    name = f"{vocab_map[hundreds_number*100]}"
                else:
                    name = f"{vocab_map[hundreds_number*100]} {vocab_map[ones_number]}"
            elif(tens_number > 0):
                if(ones_number == 0 ):
                    name = f"{vocab_map[hundreds_number*100]} {vocab_map[tens_number*10]}"
                else:
                    name = f"{vocab_map[hundreds_number*100]} {vocab_map[tens_number*10]} {vocab_map[ones_number]}"

           
        r = random.random()
        if(r<0.8):
            train_names.append(name)
            train_numbers.append(number)
        elif(r<0.9):
            val_names.append(name)
            val_numbers.append(number)
        elif(r<1.0):
            test_names.append(name)
            test_numbers.append(number)

    with open('train.txt', 'w') as f:
        for i in range(len(train_names)):
            f.write(f"{train_numbers[i]} {train_names[i]}\n")
    with open('val.txt', 'w') as f:
        for i in range(len(val_names)):
            f.write(f"{val_numbers[i]} {val_names[i]}\n")
    with open('test.txt', 'w') as f:
        for i in range(len(test_names)):
            f.write(f"{test_numbers[i]} {test_names[i]}\n")

if __name__ == '__main__':
    number_to_names()

