import random
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
  
from nltk.corpus import wordnet, stopwords

# ========================== Synonym Replacement ========================== #
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def get_random_synonym(word):
    syns = get_synonyms(word)
    if len(syns) == 0:
        return word
   
    return random.choice(syns)

def synonym_replacement(sentence, n):
    
    stop_words = set(stopwords.words('english'))
    
    words = sentence.split()
    
    ############################################################################
    # TODO: Replace up to n random words in the sentence with their synonyms.  #
    #   You should                                                             #
    #   - (i)   replace random words with one of its synonyms, until           #
    #           the number of replacement gets to n or all the words           #
    #           have been replaced;                                            #
    #   - (ii)  NO stopwords should be replaced!                               #
    #   - (iii) return a new sentence after all the replacement.               #
    ############################################################################
    # Replace "..." with your code
    words_to_sample = [word for word in words if word not in stop_words]
    if n > len(words_to_sample):
        n = len(words_to_sample)
    random_word_list = random.sample(words_to_sample, n)
    new_sentence_words = []
    for word in words:
        if word in random_word_list:# and len(get_synonyms(word))>0:
            #new_sentence_words.append(get_synonyms(word)[0])  
            new_sentence_words.append(get_random_synonym(word))
        else:
            new_sentence_words.append(word)
    
    new_sentence = " ".join(new_sentence_words)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return " ".join(words)

    ############################################################################
    # TODO: Randomly delete words with probability p. You should               #
    # - (i)   iterate through all the words and determine whether each of them #
    #         should be deleted;                                               #
    # - (ii)  you can delete at most `max_deletion_n` words;                   #
    # - (iii) return the new sentence after deletion.                          #
    ############################################################################
    # Replace "..." with your code
    new_words = []
    deletions = 0
    for word in words:
        r = random.random()
        if (r < p) and (deletions < max_deletion_n):
            deletions += 1
        else:
            new_words.append(word)
    new_sentence = " ".join(new_words)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    
    words = sentence.split()
    if len(words) <= 1:
      return sentence
    ############################################################################
    # TODO: Randomly swap two words in the sentence. You should                #
    # - (i)   randomly get two indices;                                        #
    # - (ii)  swap two tokens in these positions.                              #
    ############################################################################
    # Replace "..." with your code
    n = len(words)
    random_idx_1, random_idx_2 = random.randint(0, n-1), random.randint(0, n-1)
    
    t1 = words[random_idx_1]
    words[random_idx_1] = words[random_idx_2]
    words[random_idx_2] = t1
    
    new_sentence = " ".join(words)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence

# ========================== Random Insertion ========================== #
def random_insertion(sentence, n):
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence

def add_word(new_words):
    
    synonyms = []
    ############################################################################
    # TODO: Randomly choose one synonym and insert it into the word list.      #
    # - (i)  Get a synonym word of one random word from the word list;         #
    # - (ii) Insert the selected synonym into a random place in the word list. #
    ############################################################################
    # Replace "..." with your code
    non_stopwords = [word for word in new_words if word not in set(stopwords.words('english'))]
    if len(non_stopwords) == 0:
        return
    
    random_word = random.choice(non_stopwords)
    
    random_synonym = get_random_synonym(random_word)
    random_index = random.randint(0, len(new_words)-1)
    new_words.insert(random_index, random_synonym)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
