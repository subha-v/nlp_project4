import torch

vocab_map = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eighteen',
    19: 'nineteen',
    20: 'twenty',
    30: 'thirty',
    40: 'fourty',
    50: 'fifty',
    60: 'sixty',
    70: 'seventy',
    80: 'eighty',
    90: 'ninety',
    100: 'one hundred',
    200: 'two hundred',
    300: 'three hundred',
    400: 'four hundred',
    500: 'five hundred',
    600: 'six hundred',
    700: 'seven hundred',
    800: 'eight hundred',
    900: 'nine hundred',

}

SOS_TOKEN = 'SOS'
EOS_TOKEN = 'EOS'

#Starting and ending

words_vocab = [SOS_TOKEN] + list(vocab_map.values()) + [EOS_TOKEN]
VOCAB_SIZE = len(words_vocab)

def word_to_index(word):
    return words_vocab.index(word)


def word_to_tensor(word): #eighty maps to a tensir
    idx = word_to_index(word)
    one_hot = torch.zeros((1,VOCAB_SIZE))
    one_hot[0,idx] = 1.
    return one_hot
# nprhase = "Three hundred thirty two"
def nphrase_to_tensor(nphrase):
    nphrase_tensors = []
    words = nphrase.split(" ") # We get back a list with every thing seperated by split
    words_mod = []
    for i in range(len(words)):
        if (words[i] == "hundred"):
            item = words[i-1]+ " " + words [i] #creates three hundred
            words_mod[-1] = item #words_mod[-] is the last letter in words_mod
        else:
            words_mod.append(words[i])

    words_mod = [SOS_TOKEN] + words_mod + [EOS_TOKEN] # Adds the starting and ending to the list

    for word in words_mod:
        nphrase_tensors.append(word_to_tensor(word))
    
    return torch.stack(nphrase_tensors) # Takes individual tensors and stacks them up
        # e g [3,1,32]



