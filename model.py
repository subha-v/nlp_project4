import torch
import torch.nn as nn
from number_utils import * 

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_to_output = nn.Linear(input_dim + hidden_dim, output_dim)
        self.output_to_output = nn.Linear(hidden_dim + output_dim, output_dim)
        self.relu = nn.ReLU() # max (0,input) Rectified Linear Unit - Activation Function
        # This is just a fancy name for max(0,input). we also want it to be recognizable to take the gradient
        self.hidden_dim = hidden_dim
    
    # x = [1,39], hidden = [1,128]
    def forward(self,x,hidden):
        combo = torch.cat([x,hidden],dim=1) # Dim is the one the tensors match on 
        # concatenating input with hidden state
        # our hidden state is the glue that connects our present with the past
        # if we didnt have the hidden state , our output would only depend on the last token/input we saw
        hidden = self.input_to_hidden(combo)
        # pass hidden state through non-linear activation function
        #
        hidden = self.relu(hidden)
        # neural networks are a long 
        out = self.input_to_output(combo)
        out = self.relu(out)
        out_combo = torch.cat([out,hidden], dim=1)
        out_combo = self.output_to_output(out_combo)
        #the output is a vector of numbers that are not necessarily between 0 and 1
        #in this case we dont need probabilities
        #we need a vector of numbers that specify the meaning of numbers
        return out_combo, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_dim) # this creates a rank 2 tensor with

class Decoder(nn.Module): # this means the class Decoder inherits from nn.Module
# nn.Module is something that makes our class usuable for a model
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_to_output = nn.Linear(input_dim + hidden_dim, output_dim)
        self.output_to_output = nn.Linear(output_dim + hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1) # outputting probabilities for tokens
        self.hidden_dim = hidden_dim


    def forward(self,x,hidden):
        combo = torch.cat([x,hidden], dim=1)
        hidden = self.input_to_hidden(combo)
        out = self.input_to_output(combo)
        out_combo = torch.cat([hidden, out], dim=1)
        out = self.output_to_output(out_combo)
        out = self.log_softmax(out)
        return out, hidden

# WE ARE CONSOLIDATING STUFF

class EncoderDecoder(nn.Module):
    def __init__(self, encoder_input_dim, encoder_output_dim, decoder_input_dim, decoder_output_dim, hidden_dim =128, max_target_length=5, teacher_forcing = False):
        super().__init__()
        self.max_target_length = max_target_length
        self.encoder = Encoder(encoder_input_dim, encoder_output_dim, hidden_dim)
        self.decoder = Decoder(decoder_input_dim, decoder_output_dim, hidden_dim)
        self.teacher_forcing = teacher_forcing # to have access to this variable accross the class


    def forward(self, nphrase_tensor, number_indicies= None): # nphrase_tensor is our input
        encoder_hidden_state = self.encoder.init_hidden()
       # we want to iterate over our nphrase_tensor  for each one of the tokens
        for i in range (nphrase_tensor.shape[0]): # this gives us the number of digits like eos
            encoder_output, encoder_hidden_state = self.encoder(nphrase_tensor[i], encoder_hidden_state)
            #out_combo      #hidden
            # we dont mind that encoder_output being overridden
            #number of indicies num_indicies is our correct answe
            # r
            # we use this while training, but not predicting

        # if number of indicies is none then we are predicting, not training 
        # its automatically predicting
        # our target length when we are predicting is just the max length

        # IF NUMBER OF INDICIES IS NONE, THEN IS MAX LENGTH OTHERWISE ITS THE SHAPE
        if(number_indicies != None):
            target_length = number_indicies.shape[1]
        else:
            target_length= self.max_target_length
        
        decoder_input = digit_to_tensor(SOS_TOKEN)
        decoder_hidden_state = encoder_output

        decoder_outputs = [digit_to_tensor(SOS_TOKEN)] # Why?
        # this ist he one_hot encoding for the SOS_TOKEN => which is also equal to the known probability of one
        # so the one hot encoding of the SOS token is also the probability of the SOS token

        # WE ARE GOING TO BE RETURNING A LIST OF THE DECODER OUTPUTS
        # BECAUSE WE NEED THESE TO COMPUTE THE LOSS

        # we want to be outputting the decoder output
        # DECODER_OUTPUT IS ACTUALLY JUST A TENSOR OF PROBABILITIES (log probabilities baiscally)

        for i in range(1,target_length):
            decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state)
            decoder_outputs.append(decoder_output)

 
            if(number_indicies != None and self.teacher_forcing):
                decoder_input = digit_to_tensor(numbers_vocab[number_indicies[0,i].item()])
             #number_indicies is a rank 2 token that has a shape [1,5] that is basically a number
                # by saying .item() we get the actual number at i
                # numbers_vocab is the vocabulary list [SOS]+[1,2,3,4] etc
                # in teacher forcing we're not using the previous output to determine the next answer
                # we're just using the correct answer here
            else:
                decoder_input = output_to_digit_tensor(decoder_output)
        
        return decoder_outputs


        


# The new size is [1, 39+128]
