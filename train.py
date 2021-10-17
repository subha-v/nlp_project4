
from torch._C import LongStorageBase
from data import NumbersDataset
from word_utils import VOCAB_SIZE, nphrase_to_tensor
from number_utils import *
from model import *
from word_utils import *
import argparse



class NumberTranslator:
    def __init__(self, dataset_path, save_model_path, num_iters, hidden_dim, teacher_forcing):
        self.dataset_path = dataset_path
        self.save_model_path = save_model_path
        self.num_iters = num_iters
        self.hidden_dim = hidden_dim
        self.teacher_forcing = teacher_forcing
        self.train_data = NumbersDataset(self.dataset_path + "/train.txt")
        self.val_data = NumbersDataset(self.dataset_path + "/val.txt")
        self.max_length = 5 # 3 for digits, 2 for sos and eos
        # This is how we create the model
        self.model = EncoderDecoder(VOCAB_SIZE, self.hidden_dim, numbers_vocab_size, numbers_vocab_size, self.hidden_dim, 
        self.max_length, self.teacher_forcing)
        self.learning_rate = 0.0005
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.NLLLoss() 

    
    def train_step(self, nphrase_tensor, number_indicies):
        decoder_outputs = self.model(nphrase_tensor, number_indicies)
        # decoder outputs is a list of the output of each token
        output = []
        loss = 0 
        # we need to go through the decoder outputs and sum up the loss
        for i in range (len(decoder_outputs)):
            loss += self.loss_func(decoder_outputs[i], number_indicies[0,i].unsqueeze(0)) # we use .squieeze because we want it to be a rank 1 tensor
            number = output_to_number(decoder_outputs[i])
            #this output_to_number helps us find the index of the highest probability
            output.append(number)


        #1. Zero the gradients
        self.optimizer.zero_grad()

        # 2. Compute the gradients

        loss.backward() # we are calculating the gradients with respect to the loss

        # 3.

        self.optimizer.step()

        return output, loss.item()





    #     encoder_hidden_state = self.encoder.init_hidden()
    #    # we want to iterate over our nphrase_tensor  for each one of the tokens
    #     for i in range (nphrase_tensor.shape[0]): # this gives us the number of digits like eos
    #         encoder_output, encoder_hidden_state = self.encoder(nphrase_tensor[i], encoder_hidden_state)
    #         #out_combo      #hidden
    #         # we dont mind that encoder_output being overridden
    #     target_length = number_indicies.shape[1]
    #     decoder_input = digit_to_tensor(SOS_TOKEN)
    #     decoder_hidden_state = encoder_output
    #     output = [SOS_TOKEN]
    #     loss = 0  
    #     for i in range(1,target_length): 
    #         decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state)
    #         # decoder output is a tensor of probabilities that is [1xnumber size]
    #         number = output_to_number(decoder_output)
    #         output.append(number) # another way to say this is output+=[number]
    #         #when we train on the right answer, this makes it faster because then the model always gets fed in the right answer
    #         #this is called teacher forcing

    #         if self.teacher_forcing:
    #             # lets say our number is 235, then number_indicies => [1,5] in shape [[0,3,4,6,11]]
    #             # decoder_input => tensor([2,3,5,'EOS'])
    #             decoder_input = digit_to_tensor(numbers_vocab[number_indicies[0,i].item()]) #gives us the tensor of number indicies
    #             # decoder_input creates a tensor of the "right answers"
    #             # specifically for NLLLoss():
    #             # pass in the log p
    #             # to go from a rank 0 tensor to a rank 1 tensor use .unsqueeze(dim)
    #             #otherwise you can index into None


    #         else:
    #             decoder_input = digit_to_tensor(number)

    #         loss += self.loss_func(decoder_output, number_indicies[0,i].unsqueeze(0))

    #         if (output==EOS_TOKEN):
    #                 break

    #     self.encoder_optimizer.zero_grad()
    #     self.decoder_optimizer.zero_grad()

    #     loss.backward()
    #     self.encoder_optimizer.step() # taking a step
    #     self.decoder_optimizer.step()

    #     return output, loss.item()



               # we must predict the end of string token 
               # next stage is to call the decoder 
               # then convert output back to Number Language
               # then add that to our ooutput list which represents the full number
               # then compute the loss for the specific token and add that to a total loss
               
        # then handle the backward pass
        # updating the parameters


    def train(self):
        for i in range(self.num_iters):
            nphrase, number, nphrase_tensor, number_indices = self.train_data.get_random_sample()
            output, loss = self.train_step(nphrase_tensor, number_indices)

            if (i%1000==0):
                print(f"Iteration[{i+1}]: Training Loss = {loss:.5f}")
                print(f"Number phrase: {nphrase}, Number: {number}")
                print(f"Prediction: {output}")
                self.model.eval()
                self.validate()
                self.model.train()
                 # setting it back to training mode
                print("-" * 50)

    def predict(self, nphrase_tensor):
        # we are going to be predicting
        with torch.no_grad(): # since we are predicting we dont want the gradients to be effected
            decoder_outputs = self.model(nphrase_tensor)
            output = []
            loss = 0 
        # we need to go through the decoder outputs and sum up the loss

        for i in range (len(decoder_outputs)):
            number = output_to_number(decoder_outputs[i])
            #this output_to_number helps us find the index of the highest probability
            output.append(number)

        return output



        
    def validate(self):
        num_correct = 0 
        total = 0 
        for i in range(len(self.val_data)): # this gives us self.size bc thats what we implemented
            nphrase, number, nphrase_tensor, number_indicies = self.val_data[i]
            pred_number_lst = self.predict(nphrase_tensor)
            pred_number_indicies = [numbers_vocab.index(elem) for elem in pred_number_lst] # this is called list comprehension
            #this gives numbers_vocab(elem) for each elem in pred_number_list
            all_correct = True
            # if we have a 1x5 tensor but the length is not correct then its not corect
            if(len(pred_number_indicies) != number_indicies.shape[1]):
                all_correct = False
            else:
                for i in range(number_indicies.shape[1]):
                    all_correct = all_correct and (pred_number_indicies[i] == number_indicies[0,i])
                    # all correct is true this is another way of implementing it
                
            if(all_correct): # if all_correct = True
                num_correct+=1
            
            total+=1
        percent_correct = (num_correct/total)*100
        print(f"Validation Accuracy: {percent_correct}%, Num_Correct: {num_correct}, Total: {total}")




if __name__ == '__main__':
    number_translator = NumberTranslator("data", "model.pth", 100000, 128, True)
    number_translator.train()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", action = "store_true")
    # parser.add_argument("--continue_training", action = "store_true")
    # parser.add_argument("--dataset_path", type = str, default = "model.pth")
    # parser.add_argument("--save_model_path", type=str )

