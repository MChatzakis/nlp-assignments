from a3_decoding import *
from a3_sampling import *

###########################################################################
# NOTE: Caution - do not modify this file!!
###########################################################################

def greedy_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int
):
    # 1) Load the decoder
    print("-" * 50)
    print("Greedy Tests")
    print("-" * 50)
    greedy_decoder = GreedySearchDecoderForT5(model=model, tokenizer=tokenizer)

    # NOTE: always do early stopping, the way describe in the decoder skeleton
    #       you don't need to implement the other ways huggingface handles it 
    for title, inputs in all_inputs:
        print("#" * 20)
        print("Input: ", title)
        print("#" * 20)

        print("~ Your Implementation ~")
        result_ids = greedy_decoder.search(
            inputs=inputs,
            max_new_tokens=max_new_tokens
        )
        if result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        print("Generated sequence: ", tokenizer.batch_decode(result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", result_ids.shape)

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        hf_result_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            num_beams=1,
            length_penalty=0.0,
            early_stopping=True,
        )
        print("Generated sequence: ", tokenizer.batch_decode(hf_result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", hf_result_ids.shape)
        print("\n")


def beam_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: int,
    num_return_sequences: int,
):
    # 1) Load the decoder
    print("-" * 50)
    print("Beam Tests")
    print("-" * 50)
    beam_decoder = BeamSearchDecoderForT5(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("#" * 20)
        print("Input: ", title)
        print("#" * 20)

        print("~ Your Implementation ~")
        result_dict = beam_decoder.search(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
        )
        if result_dict is None:
            print("Input constraint encountered. Exiting...")
            exit()
        seq_scores = enumerate(zip(result_dict["scores"], tokenizer.batch_decode(result_dict["sequences"], skip_special_tokens=False)))
        for i, (score, seq) in seq_scores:
            print("{}. score: {}".format(i + 1, score))
            print("{}. generated sequence: {}".format(i + 1, seq))
            #print("{}. generated sequence ids: {}".format(i + 1, result_dict["sequences"][[i]]))
        print("Best output shape: ", result_dict["sequences"].shape)

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        hf_result_dict = model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            num_beams=num_beams,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            return_dict_in_generate=True,
            num_beam_groups=1,
            constraints=None,
            output_scores=True
        )
        seq_scores = enumerate(zip(hf_result_dict["sequences_scores"], tokenizer.batch_decode(hf_result_dict["sequences"], skip_special_tokens=False)))
        for i, (score, seq) in seq_scores:
            print("{}. score: {}".format(i + 1, score))
            print("{}. generated sequence: {}".format(i + 1, seq))
        print("Best output shape: ", hf_result_dict["sequences"].shape)
        print("\n")
        

def top_k_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int
):
    # 1) Load the decoder
    print("-" * 50)
    print("Top-k Tests")
    print("-" * 50)
    top_k_sampler = TopKSamplerForT5(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("#" * 20)
        print("Input: ", title)
        print("#" * 20)

        print("~ Your Implementation ~")
        torch.manual_seed(seed)
        result_ids = top_k_sampler.sample(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature
        )
        if result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        print("Generated sequence: ", tokenizer.batch_decode(result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", result_ids.shape)

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(seed)
        hf_result_ids = model.generate(
            **inputs, 
            do_sample=True, 
            num_beams=1,
            max_new_tokens=max_new_tokens, 
            length_penalty=0.0,
            early_stopping=True,
            top_k=top_k,
            temperature=temperature
        )
        print("Generated sequence: ", tokenizer.batch_decode(hf_result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", hf_result_ids.shape)
        print("\n")


def top_p_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    seed: int
):
    # 1) Load the decoder
    print("-" * 50)
    print("Top-p Tests")
    print("-" * 50)
    top_p_sampler = TopPSamplerForT5(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("#" * 20)
        print("Input: ", title)
        print("#" * 20)

        print("~ Your Implementation ~")
        torch.manual_seed(seed)
        result_ids = top_p_sampler.sample(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        if result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        print("Generated sequence: ", tokenizer.batch_decode(result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", result_ids.shape)

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(seed)
        hf_result_ids = model.generate(
            **inputs, 
            do_sample=True, 
            num_beams=1,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            length_penalty=0.0,
            top_p=top_p,
            top_k=0, # deactivate top_k sampling
            temperature=temperature,
        )
        print("Generated sequence: ", tokenizer.batch_decode(hf_result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", hf_result_ids.shape)
        print("\n")