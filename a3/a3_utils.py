import os
import torch
import random
import numpy as np
import transformers

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

MAX_T5_SEQ_LENGTH = 512

class GeneratorForT5():
    ###########################################################################
    # NOTE: Caution - do not modify the inputs to the helper class, however, feel free
    # to add as many helper functions in this class as you want or to modify the 
    # prepare_decoder_input_ids function :)
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        """A helper generator class for decoding and sampling algorithms.
        You are free to not use its fields or the self.prepare_next_inputs function.

        Args:
            model (T5ForConditionalGeneration): model to generate with
            tokenizer (T5Tokenizer): corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_model_length = MAX_T5_SEQ_LENGTH
        self.eos_token_id = model.config.eos_token_id
        self.pad_token_id = model.config.pad_token_id
        self.model.eval()

    def input_constraints(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams : int = None,
        num_return_sequences: int = None,
        top_k: int = None
    ):
        """A helper function to let you know that you don't need to handle the 
        certain edge cases.

        Args:
            inputs (dict)
            max_new_tokens (int)
            num_beams (int, optional). Defaults to None.
            num_return_sequences (int, optional). Defaults to None.

        Returns:
            Any: either max_new_tokens or None if not within constraints
        """
        if max_new_tokens < 1:
            print("Generation should be at least 1 token. Returning None.")
            return None
        batch_size = inputs["input_ids"].shape[0]
        if batch_size != 1:
            print(f"Your batch_size={batch_size} but this function only handles batch_size=1. Returning None.")
            return None
        if self.max_model_length < max_new_tokens:
            print("Truncating max_new_tokens = {} to the model's maximum capacity = {}.".format(
                max_new_tokens,
                self.max_model_length))
            max_new_tokens = self.max_model_length
            return max_new_tokens
        # Only concerns beam search
        if num_return_sequences is not None:
            if num_return_sequences > num_beams or num_return_sequences < 2:
                print("num_return_sequences should be more than 1 and less than num_beams.")
                return None
        if top_k is not None:
            if top_k < 1:
                print("top_k should be more than 0.")
                return None
        
        # Otherwise return original max_new_tokens
        return max_new_tokens

    def prepare_next_inputs(
            self,
            model_inputs: dict,
            new_token_id: torch.Tensor = None,
            use_cuda: bool = False
        ) -> dict:
        """"A helper function to prepare decoder input ids and their attention mask 
        to be passed during the forward pass of the model. 

        You do not need to use this function + feel free to modify it!

        Args:
            model_inputs (dict): the last inputs to the model
            new_token_id (torch.Tensor, optional): the token ID to be added to old decoder inputs. Defaults to None. Defaults to None.
            returned_past_key_values (torch.Tensor, optional): cached past_key_values. You don't need to use them. Defaults to None.
            use_cuda (bool): Whether to move tensors to cuda or not. Defaults to False.

        Returns:
            dict: the next model input dictionary
        """
        new_model_inputs = {}
        if new_token_id is None:
            # First step of decoding, we need to create a tensor of decoder input ids, decoder attention mask
            new_model_inputs.update(model_inputs)
            new_model_inputs["decoder_input_ids"] = torch.zeros(size=(1, 1), dtype=torch.long)
            new_model_inputs["decoder_input_ids"].fill_(value=self.pad_token_id)    
            if use_cuda:
                new_model_inputs["decoder_input_ids"] = new_model_inputs["decoder_input_ids"].cuda()
            new_model_inputs["decoder_attention_mask"] = torch.ones(size=(1, 1))
            if use_cuda:
                new_model_inputs["decoder_attention_mask"] = new_model_inputs["decoder_attention_mask"].cuda()
            new_model_inputs["past_key_values"] = None
        else:
            # Next steps of decoding, we need to concatenate the new token + attention mask
            new_model_inputs["decoder_input_ids"] = torch.cat([
                model_inputs["decoder_input_ids"],
                model_inputs["decoder_input_ids"].new_ones((model_inputs["decoder_input_ids"].shape[0], 1))],
                dim=-1,
            )
            new_model_inputs["decoder_input_ids"][0][-1] = new_token_id
            new_model_inputs["decoder_attention_mask"] = torch.cat([
                model_inputs["decoder_attention_mask"], 
                model_inputs["decoder_attention_mask"].new_ones((model_inputs["decoder_attention_mask"].shape[0], 1))
            ],dim=-1)
            new_model_inputs["input_ids"] = model_inputs["input_ids"]
            new_model_inputs["attention_mask"] = model_inputs["attention_mask"]
            # NOTE: don't need to do this if decoder_input_ids is the full input
            # new_model_inputs["past_key_values"] = returned_past_key_values
        return new_model_inputs


def load_seed(seed : int):
    """Sets the seed for several packages you may use, in case it's not torch.
    You can also just do torch.manual_seed(seed).

    Args:
        seed (int): the seed number
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: if distributed training/inference
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    
    
def shuffle_tensor(tensor : torch.Tensor):
    """Shuffles a tensor along its first dimension.

    Args:
        tensor (torch.Tensor): the tensor to shuffle
        seed (int): the seed to use for shuffling

    Returns:
        torch.Tensor: the shuffled tensor
    """

    shuffled_tensor = tensor[torch.randperm(tensor.shape[0])]
    return shuffled_tensor

def check_if_sequences_have_ended(sequences: list, pad_id: int, eos_id: int, num_return_seqs: int ,printd):
    """_summary_

    Args:
        sequences (list): _description_
        pad_id (int): _description_
        eos_id (int): _description_

    Returns:
        _type_: _description_
    """
    for index, sequence in enumerate(sequences[:num_return_seqs]):
        last_token_id = sequence[-1]
        
        if printd:
            print(last_token_id)
        
        if(last_token_id != pad_id and last_token_id != eos_id):
            return False
        
    return True

def trim_sentences(sequences: list, pad_id: int, eos_id: int):
    """_summary_

    Args:
        sequences (_type_): _description_
        pad_id (_type_): _description_
        eos_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    reversed_sequences = []
    for seq in sequences:
        reversed_seq = seq[::-1]
        reversed_sequences.append(reversed_seq)
        
    possible_deletions = 0
    for index in range(len(reversed_sequences[0])):
        # Check if the first token of any list in reversed_sequences is the same
        if(check_same_index(reversed_sequences, index, pad_id)):
            possible_deletions += 1
        else:
            break

    trimmed_sequences = []
    for seq in sequences:
        trimmed_sequences.append(seq[:len(seq)-possible_deletions])
    
    return trimmed_sequences
    
def check_same_index(sequences: list, index: int, pad_id: int):
    """_summary_

    Args:
        sequences (_type_): _description_
        index (_type_): _description_
        pad_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    for seq in sequences:
        if(seq[index] != pad_id):
            return False
        
    return True
