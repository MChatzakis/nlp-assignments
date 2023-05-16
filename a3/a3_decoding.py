import torch
from typing import Any, Dict
from a3_utils import *
from sys import exit
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math


class GreedySearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of
    # the sample function.
    #
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)

    def search(self, inputs: dict, max_new_tokens: int) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration
        (which has a language modeling head) using greedy decoding.
        This means that we always pick the next token with the highest score/probability.

        This function always does early stopping and does not handle the case
        where we don't do early stopping.
        It also only handles inputs of batch size = 1.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs
                                  we desire to generate

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
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
        # For greedy decoding, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
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

        # We do not handle cases where batch size is not 1
        if len(inputs["input_ids"]) != 1:
            print("GreedySearch function handles only inputs with batch size 1.")
            print(f"The batch size of the given input is {len(inputs['input_ids'])}")
            print("Returning None.")
            return None

        # Pad token is also used as start token
        start_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id

        # Initialize the sequence with the start token
        sequence_ids = [start_token_id]

        model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        for token_num in range(max_new_tokens):
            model_outputs = self.model(**model_inputs)

            # Get the logits for the next token and convert them to probabilities
            token_prob_logits = model_outputs.logits[:, -1, :]
            token_prob_distribution = torch.nn.functional.softmax(
                token_prob_logits, dim=-1
            )

            # Get the token id with the highest probability
            max_prob_token_idx = torch.argmax(token_prob_distribution, dim=1)
            new_token_id = max_prob_token_idx.item()
            # print(f'Token ID with highest probability: {new_token_id}')

            # Prepare the next input
            model_inputs = self.prepare_next_inputs(
                model_inputs=model_inputs,
                new_token_id=new_token_id,
            )

            # Add the new token to the sequence
            sequence_ids.append(new_token_id)

            # Stop decoding if we reach the end token: Early stopping
            if new_token_id == end_token_id:
                break

        # Convert the list of token ids to a long tensor, with the required shape
        result = torch.tensor(sequence_ids, dtype=torch.int64).view(1, -1)
        return result


class BeamSearchDecoderForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of
    # the sample function.
    #
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)

    def search(
        self,
        inputs,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0,
    ) -> dict:
        """Generates sequences of token ids for T5ForConditionalGeneration
        (which has a language modeling head) using beam search.
        This means that we sample the next token according to the best conditional
        probabilities of the next beam_size tokens.

        This function always does early stopping and does not handle the case
        where we don't do early stopping.
        It also only handles inputs of batch size = 1 and of beam size > 1
            (1=greedy search, but you don't have to handle it)

        It also include a length_penalty variable that controls the score assigned to a long generation.
        Implemented by exponiating the length of the decoder inputs to this value.
        This is then used to divide the score which can be calculated as the sum of the log probabilities so far.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (_type_): the tokenized input dictionary returned by the T5 tokenizer
            max_new_tokens (int): a limit for the amount of decoder outputs
                                  we desire to generate
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional):
                exponential penalty to the length that is used with beam-based generation.
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor)
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence,
                          you can fill the rest of the tensor row with the pad token ID
                    - "scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs,
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # Given a probability distribution over the possible next tokens and
        # a beam width (here num_beams), needs to keep track of the most probable
        # num_beams candidates.
        # You can do so by keeping track of the sum of the log probabilities of
        # the best num_beams candidates at each step.
        # Then recursively repeat this process until either:
        #   - you reach the end of the sequence
        #   - or you reach max_length
        #
        # For beam search, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the length penalty
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

        # Initialize special token variables
        start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id

        # Set this to True if you want to print out the beam search steps
        DEBUG = False

        # We do not handle cases where batch size is not 1
        if len(inputs["input_ids"]) != 1:
            print("BeamSearch function handles only inputs with batch size 1.")
            print(f"The batch size of the given input is {len(inputs['input_ids'])}")
            print("Returning None.")
            return None

        # We do not handle cases where num_beams is 1, we provide Greedy Search function for that
        if num_beams == 1:
            print("BeamSearch with 1 beam is the same as greed search.")
            print("Returning None.")
            return None

        # Dictionary to store the sequences and their respective scores
        # The size of the dictionary is equal to the number of beams
        sequences_dict = {"sequences": [], "scores": [], "model_inputs": [], "penalized_scores": []}

        model_inputs = self.prepare_next_inputs(model_inputs=inputs)

        for step in range(max_new_tokens):
            # For the first step, we do special handling
            # We initialize the sequences_dict with the start token and the first token
            if step == 0:
                # Get top-(num_beams) tokens and their probabilities
                token_ids, vals = self.get_token_ids_with_probabilities(
                    model_inputs, num_beams
                )

                # Add the start token and the first token to the sequences_dict with the initial scores,
                # as well as the model_inputs
                for index, token_id in enumerate(token_ids[0]):
                    token_id_value = token_id.item()
                    sequences_dict["sequences"].append([start_token_id, token_id_value])
                    
                    sequences_dict["scores"].append(torch.log(vals[0][index]).item())
                    sequences_dict["penalized_scores"].append(torch.log(vals[0][index]).item())
                    
                    sequences_dict["model_inputs"].append(model_inputs)
            # For the rest of the steps, we do the following:
            else:
                # Gathers the scores of all sequences that the current sequences will generate
                # It is used to calculate the next best num_beams sequences
                all_results = []

                # Iterate over all candidate sequences in the sequences_dict
                for seq_index in range(num_beams):
                    # Get the last token of the current iteration's sequence
                    previous_token_id = sequences_dict["sequences"][seq_index][-1]

                    # Get previous model inputs for this sequence
                    previous_partial_model_inputs = sequences_dict["model_inputs"][
                        seq_index
                    ]

                    # In this sequence has finished, we add pad token and add
                    # it to the all_results list with the same score
                    if (
                        previous_token_id == end_token_id
                        or previous_token_id == pad_token_id
                    ):
                        seq = sequences_dict["sequences"][seq_index] + [pad_token_id]
                        score = sequences_dict["scores"][seq_index]
                        penalized_scores = sequences_dict["penalized_scores"][seq_index]

                        # Append the sequence and relevant data to all_results list
                        all_results.append((penalized_scores, seq, previous_partial_model_inputs, score))
                    else:
                        # Get the next model inputs for this sequence
                        partial_model_inputs = self.prepare_next_inputs(
                            model_inputs=previous_partial_model_inputs,  # model_inputs
                            new_token_id=previous_token_id,
                        )

                        # Get top-(num_beams) tokens and their probabilities
                        token_ids, vals = self.get_token_ids_with_probabilities(
                            partial_model_inputs, num_beams
                        )

                        # For each newly generated sequence, calculate the new score and add them to all_results
                        for index, token_id in enumerate(token_ids[0]):
                            # Get the value of the token
                            token_id_value = token_id.item()

                            # Add it to the corresponding sequence
                            seq = sequences_dict["sequences"][seq_index] + [
                                token_id_value
                            ]

                            # Calculate the overall score with penalty
                            original_score = (
                                sequences_dict["scores"][seq_index]
                                + torch.log(vals[0][index]).item()
                            )

                            score = original_score / ((len(seq)) ** length_penalty)

                            # Append the sequence and relevant data to all_results list
                            all_results.append((score, seq, partial_model_inputs, original_score))

                # Sort the all_results list by score and get the top-(num_beams) results
                sorted_results = sorted(
                    all_results, key=lambda tup: tup[0], reverse=True
                )  # by score
                top_beam_results = sorted_results[:num_beams]

                # Register top results to final dict.
                for index, result in enumerate(top_beam_results):
                    sequences_dict["penalized_scores"][index] = result[0]
                    sequences_dict["sequences"][index] = result[1]
                    sequences_dict["model_inputs"][index] = result[2]
                    sequences_dict["scores"][index] = result[3] # We keep the original score for the next step

            # Print the state of the data after each step if DEBUG is enabled
            if DEBUG:
                print(f"State of data in the end of {step} Beam Step:")
                print("Sequences:", sequences_dict["sequences"])
                print("Scores:", sequences_dict["scores"])
                print("Penalized Scores:", sequences_dict["penalized_scores"])
                print("============\n")
                
            # Check if early stopping is needed
            if(check_if_sequences_have_ended(sequences_dict["sequences"], pad_token_id, end_token_id, num_beams, DEBUG)):
                break
        
        # Apply trimming patch for uneeded pad tokens
        sequences_dict["sequences"] = trim_sentences(sequences_dict["sequences"][:num_return_sequences], pad_token_id, end_token_id)
    
        # Return the top beam result as a tensor
        result = {}
        result["sequences"] = torch.tensor(sequences_dict["sequences"][:num_return_sequences])
        result["scores"] = sequences_dict["penalized_scores"][:num_return_sequences]
        return result

    def get_token_ids_with_probabilities(self, model_inputs, num_beams):
        """This function returns the top-(num_beams) tokens and their probabilities

        Args:
            model_inputs (_type_): Inputs of the model
            num_beams (_type_): The number of beams

        Returns:
            list, list: the token ids and their probabilities
        """
        model_outputs = self.model(**model_inputs)
        logits = model_outputs.logits[:, -1, :]

        token_prob_distribution = torch.nn.functional.softmax(logits, dim=-1)
        # print(f"Distribution: {token_prob_distribution}, with size: {token_prob_distribution.size()}")

        _, token_ids = torch.topk(token_prob_distribution, k=num_beams)
        # print(f"TokenIDs: {token_ids}, with size: {token_ids.size()}")

        vals = token_prob_distribution[:, token_ids.squeeze()]
        # print(f"Probs of the corresponding tokens: {vals} with size {vals.size()}")
        return token_ids, vals


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


if __name__ == "__main__":
    main()
