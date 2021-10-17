import torch


SOS_TOKEN = 'SOS'
EOS_TOKEN = 'EOS'
numbers_vocab =[SOS_TOKEN] + [0,1,2,3,4,5,6,7,8,9]+ [EOS_TOKEN]

numbers_vocab_size =  len(numbers_vocab)

def digit_to_index(digit):
    return numbers_vocab.index(digit)

def digit_to_tensor(digit):
    one_hot = torch.zeros((1,numbers_vocab_size))
    idx = digit_to_index(digit)
    one_hot[0, idx] = 1
    return one_hot

def number_to_other(number, is_tensor= False):

    apply_func = digit_to_index
    if (is_tensor):
        apply_func = digit_to_tensor

    ret = [] # Generic value for the list
    ret.append(apply_func(SOS_TOKEN))
    
    if(number < 10):
        ret.append(apply_func(number))
    elif(number < 100):
        ret.append(apply_func((number % 100 ) // 10 ))
        ret.append(apply_func(number % 10))
    elif(number < 1000):
        ret.append(apply_func(number //100))
        ret.append(apply_func((number % 100) // 10))
        ret.append(apply_func(number % 10))
    ret.append(apply_func(EOS_TOKEN))
    
    if(is_tensor):
        return torch.stack(ret) # 
    
    return ret

def number_to_tensor(number):
    return number_to_other(number, is_tensor=True)

def number_to_indices(number):
    return number_to_other(number)
# More robust code


# indicies are like a list [0,3,1,2,11]
def indices_to_number(indicies):
    # List comprehension
    return [numbers_vocab[i] for i in indicies]  # Creating a list

def output_to_number(output):
    # [1,12] tensor
    idx = output.argmax(dim=1)[0].item()
    return numbers_vocab[idx]

def output_to_digit_tensor(output):
    # we just go automatically to the one hot
    return digit_to_tensor(output_to_number(output))


