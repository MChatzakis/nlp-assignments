import itertools
import jsonlines
import nltk

import matplotlib.pyplot as plt
import numpy as np

import string

nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("english")
stop_words.append("uh")

puncs = string.punctuation


def word_pair_extraction(prediction_files, tokenizer):
    """
    Extract all word pairs (word_from_premise, word_from_hypothesis) from input as features.

    INPUT:
      - prediction_files: file path for all predictions
      - tokenizer: tokenizer used for tokenization

    OUTPUT:
      - word_pairs: a dict of all word pairs as keys, and label frequency of values.
    """
    word_pairs = {}
    label_to_id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            for pred in reader.iter():
                #########################################################
                #          TODO: construct word_pairs dictionary        #
                #  - tokenize the text with 'tokenizer'                 #
                #  - pair words as keys (you can use itertools)         #
                #  - count predictions for each paired words as values  #
                #  - remenber to filter undesired word pairs            #
                #########################################################
                # Replace "..." statement with your code

                # ...

                # 1. Clean up text from punctuation and stop words and get IDs
                label = pred["prediction"]

                premise_clean = clean_punc(pred["premise"], puncs).lower()
                hypothesis_clean = clean_punc(pred["hypothesis"], puncs).lower()

                premise_tokens = filter_tokens(
                    tokenizer.tokenize(premise_clean), stop_words
                )
                hypothesis_tokens = filter_tokens(
                    tokenizer.tokenize(hypothesis_clean), stop_words
                )

                premise_ids = tokenizer.convert_tokens_to_ids(premise_tokens)
                hypothesis_ids = tokenizer.convert_tokens_to_ids(hypothesis_tokens)
                label_id = label_to_id[label]

                # 2. Get all combination pairs and filter pairs that have duplicates
                combination_pairs = list(itertools.product(premise_ids, hypothesis_ids))
                combination_pairs = [
                    pair for pair in combination_pairs if pair[0] != pair[1]
                ]

                for p in combination_pairs:
                    assert p[0] != p[1]

                    counters = None

                    if p in word_pairs:
                        counters = word_pairs[p]
                        counters[label_id] += 1
                    else:
                        counters = [0, 0, 0]
                        counters[label_id] += 1
                    word_pairs[p] = counters

                # print(combination_pairs)

                #####################################################
                #                   END OF YOUR CODE                #
                #####################################################

    return word_pairs


def clean_punc(text, puncs):
    translator = str.maketrans("", "", puncs)
    clean_text = text.translate(translator)
    return clean_text


def filter_tokens(token_list, stopwords, fprefix="##"):
    filtered = [
        token
        for token in token_list
        if ((token not in stopwords) and (not token.startswith(fprefix)))
    ]

    return filtered


def get_representatives(
    prediction_files,
    pair_data,
    tokenizer,
    fail_classify=True,
    num_examples=2,
    fail_label=True,
):
    examples = []

    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            for pred in reader.iter():
                gold_label = pred["label"]
                model_prediction = pred["prediction"]

                premise_clean = clean_punc(pred["premise"], puncs).lower()
                hypothesis_clean = clean_punc(pred["hypothesis"], puncs).lower()

                premise_tokens = filter_tokens(
                    tokenizer.tokenize(premise_clean), stop_words
                )
                hypothesis_tokens = filter_tokens(
                    tokenizer.tokenize(hypothesis_clean), stop_words
                )
                p_premise = pair_data[0]
                p_hypothesis = pair_data[1]
                classification_type = pair_data[2]

                if p_premise in premise_tokens and p_hypothesis in hypothesis_tokens:
                    if fail_classify == True:
                        if (
                            classification_type == model_prediction
                            and model_prediction != gold_label
                        ):
                            examples.append(
                                pred
                            )  # Add only if it is missclassified from the model
                    elif fail_label == True:
                        if gold_label != classification_type:
                            examples.append(pred)
                    else:
                        examples.append(pred)

                    if len(examples) == num_examples:
                        break

        if len(examples) == num_examples:
            break

    return examples


def plot_all(entail, contr):
    fig, ax = plt.subplots(1, 2)

    N = 5
    ind = np.arange(N)
    width = 0.25

    ent_vals = []
    net_vals = []
    contr_vals = []
    labels = []
    for item in entail:
        ent = item[3][0]
        ent_vals.append(ent)

        net = item[3][1]
        net_vals.append(net)

        con = item[3][2]
        contr_vals.append(con)

        labels.append(item[0] + "-" + item[1])

    bar1 = ax[0].bar(ind, ent_vals, width, color="g")
    bar2 = ax[0].bar(ind + width, net_vals, width, color="b")
    bar3 = ax[0].bar(ind + 2 * width, contr_vals, width, color="r")

    ax[0].set_xticks(ind + width, labels)
    # ax[0].legend((bar1, bar2, bar3), ('Entailment', 'Neutral', 'Contradiction'))
    ax[0].tick_params(axis="both", which="major", labelsize=10, rotation=90)
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel("Word Pairs")
    ax[0].set_title("Entailment pairs")

    ent_vals = []
    net_vals = []
    contr_vals = []
    labels = []
    for item in contr:
        ent = item[3][0]
        ent_vals.append(ent)

        net = item[3][1]
        net_vals.append(net)

        con = item[3][2]
        contr_vals.append(con)

        labels.append(item[0] + "-" + item[1])

    bar1 = ax[1].bar(ind, ent_vals, width, color="g")
    bar2 = ax[1].bar(ind + width, net_vals, width, color="b")
    bar3 = ax[1].bar(ind + 2 * width, contr_vals, width, color="r")

    ax[1].set_xticks(ind + width, labels)
    ax[1].legend((bar1, bar2, bar3), ("Entailment", "Neutral", "Contradiction"))
    ax[1].tick_params(axis="both", which="major", labelsize=10, rotation=90)
    ax[1].set_ylabel("Count")
    ax[1].set_xlabel("Word Pairs")
    ax[1].set_title("Contradiction pairs")

    fig.suptitle("Shortcut word pairs")
    fig.tight_layout()
    plt.show()

    return
