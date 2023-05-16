import torch
from typing import Any, Dict
from a3_utils import *

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


class TopKSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        It also only handles top_k => 1.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_k (int): the number of highest probability vocabulary tokens to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-k sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        
        # We only handle batches of size 1
        if len(inputs["input_ids"]) != 1:
            print("BeamSearch function handles only inputs with batch size 1.")
            print(f"The batch size of the given input is {len(inputs['input_ids'])}")
            print("Returning None.")
            return None
        
        # We only handle top_k >= 1
        if top_k < 1:
            print(f"Tok-K function handles only tok-k values >=1.")
            print(f"Current k value: {top_k}")
            print("Returning None.")
            return None
        
        if temperature == 0:
            print("Warning: Running with Temperature=0.")
        
        # Set start and end token IDs
        start_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id

        # Initialize the sequence of token IDs
        sequence_ids = [start_token_id]

        model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        for token_num in range(max_new_tokens):
            # Get model outputs
            model_outputs = self.model(**model_inputs)

            # Get the logits for the last token and scale them based on temperature
            token_logits = model_outputs.logits[:, -1, :]
            token_logits_scaled = token_logits/temperature if temperature != 0 else token_logits
            
            # Zeroing the probabilities of the tokens we don't want to sample from
            # Same method as HuggingFace, to get the same results
            indices_to_remove = token_logits_scaled < torch.topk(token_logits_scaled, top_k)[0][..., -1, None] # reshape the tensor in the end
            token_logits_scaled[indices_to_remove] = -float('Inf') 
            
            # Check for temperature value, if 0 we mean greedy sampling
            if temperature == 0: 
                # Get top-probability token (Greedy Search)
                new_token_id = torch.argmax(token_logits_scaled, dim=-1).unsqueeze(-1)
            else:
                # Draw 1 sample from the top-k distribution
                new_token_id = torch.multinomial(torch.nn.functional.softmax(token_logits_scaled, dim=-1), num_samples=1)
            
            #print(f'Sampled token ID: {new_token_id}')

            # Update model inputs
            model_inputs = self.prepare_next_inputs(
                model_inputs=model_inputs,
                new_token_id=new_token_id,
            )

            # Update sequence of token IDs
            sequence_ids.append(new_token_id)

            # Early stopping if we encounter the end token
            if new_token_id == end_token_id:
                break

        # Convert the sequence of token IDs to a tensor to return it in the correct format
        result = torch.tensor(sequence_ids, dtype=torch.int64).view(1, -1)

        return result

    

class TopPSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p or higher.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) then sample the top scoring token.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-p sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to handle the edge case when top scoring tokens probability > top_p,
        #     sample that token only.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        
        # We do not handle input batch size != 1. 
        if len(inputs["input_ids"]) != 1:
            print("BeamSearch function handles only inputs with batch size 1.")
            print(f"The batch size of the given input is {len(inputs['input_ids'])}")
            print("Returning None.")
            return None
        
        if temperature == 0:
            print("Warning: Running with Temperature=0.")
        
        # Initialize the sequence of token IDs and sequence IDs
        start_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id
        sequence_ids = [start_token_id]

        # Initialize model inputs
        model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        
        # Loop over the maximum number of tokens we want to generate
        for token_num in range(max_new_tokens):
            # Get model outputs
            model_outputs = self.model(**model_inputs)

            # Get the logits of the next token, scale them by temperature and apply softmax
            token_logits = model_outputs.logits[:, -1, :]
            token_logits_scaled = token_logits/temperature if temperature != 0 else token_logits

            # Get the top-p logits
            filtered_logits = self.get_top_p_logits_v1(token_logits_scaled, top_p) # v1 for same results, v2 for approx. same results
            
            if temperature == 0: # greedy sampling:
                new_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                new_token_id = torch.multinomial(torch.nn.functional.softmax(filtered_logits, dim=-1), num_samples=1)
            
            #print(f'Token ID : {new_token_id}')

            # Update model inputs
            model_inputs = self.prepare_next_inputs(
                model_inputs=model_inputs,
                new_token_id=new_token_id,
            )

            # Update sequence IDs with the new token ID
            sequence_ids.append(new_token_id)

            # If the new token ID is the end token ID, break the loop: Early stopping
            if new_token_id == end_token_id:
                break

        # Convert the sequence of token IDs to a tensor in the needed format
        result = torch.tensor(sequence_ids, dtype=torch.int64).view(1, -1)
        return result
    
    def get_top_p_logits_v1(self, logits, top_p):
        # Sort the logits to have the one with highest value first
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute the cumulative probabilities of the tokens for top p
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map the indices to their original positions
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        
        # Set the logit of all tokens that wont be selected to -Inf so they have zero probability
        filtered_logits = logits.clone() # Clone to avoid modifying the original tensor
        filtered_logits[indices_to_remove] = -float('Inf') # Zeroing the probability
        
        return filtered_logits
    
    def get_top_p_logits_v2(self, logits, top_p):
        filtered_logits = logits.clone()
        
        # Get tokens sorted by their probability in descending order
        token_prob_distribution = torch.nn.functional.softmax(logits, dim=-1)
        _, token_ids = torch.topk(token_prob_distribution, k=len(token_prob_distribution))

        # Cumulative sum of the probabilities of the top scoring tokens
        curr_prob_sum = 0.0
        
        # Loop over the sorted tokens
        for token_id in token_ids[0]:
            # If the cumulative sum of the probabilities is greater than top_p, we "remove" the token
            if curr_prob_sum > top_p:
                filtered_logits[0][token_id] = -float('Inf')       
            
            tok_prob = token_prob_distribution[0][token_id]
            curr_prob_sum += tok_prob     
        
        return filtered_logits
        
        
def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()