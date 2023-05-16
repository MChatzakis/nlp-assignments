from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn
import random


class RNNDataset(Dataset):
    def __init__(self, dataset: datasets.arrow_dataset.Dataset, max_seq_length: int):
        # YOUR CODE HERE
        self.max_seq_length = max_seq_length + 2  # Including <start> and <stop>

        self.prepared_dataset = self.add_start_stop_tokens_per_sample(
            dataset
        )  # Will add <start> and <stop> tokens per sample

        # Defining a dictionary that simply maps tokens to their respective index in the embedding matrix
        self.vocab = self.get_dataset_vocabulary(self.prepared_dataset)

        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        # Saving the ID of <pad> token
        self.pad_idx = self.word_to_index["<pad>"]

    def __len__(self):
        return len(self.prepared_dataset)

    def __getitem__(self, idx):
        # here we need to transform the data to the format we expect at the model input
        token_list = self.prepared_dataset[idx].split()

        # having a fallback to <unk> token if an unseen word is encoded.
        token_ids = [
            self.word_to_index.get(word, self.word_to_index["<unk>"])
            for word in token_list
        ]

        # Padded tensor
        return torch.tensor(
            token_ids + [self.pad_idx] * (self.max_seq_length - len(token_list))
        )

    def get_dataset_vocabulary(self, dataset):
        vocab = sorted(set(" ".join([sample for sample in dataset]).split()))
        vocab += ["<pad>"]  # Manually add the padding token
        return vocab

    def add_start_stop_tokens_per_sample(self, raw_dataset):
        prepared_dataset = []
        for sample in raw_dataset:
            prepared_dataset.append("<start> " + sample["text"] + " <stop>")

        return prepared_dataset

    def decode_idx_to_word(self, token_id):
        return [self.index_to_word[id_.item()] for id_ in token_id]

    def get_encoded_dataset_samples(self):
        all_token_lists = [sample.split() for sample in self.prepared_dataset]

        # padding every sentence to max_seq_length
        all_token_ids = [
            [
                self.word_to_index.get(word, self.word_to_index["<unk>"])
                for word in token_list
            ]
            + [self.pad_idx] * (self.max_seq_length - len(token_list))
            for token_list in all_token_lists
        ]

        return torch.tensor(all_token_ids)


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        embedding_weights=None,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights, freeze=freeze_embeddings
            )
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # YOUR CODE HERE
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, context, criterion):
        context = context.t()  # transposing it for LSTM model

        vector = self.embedding(context)

        embedded = self.dropout(vector)

        outputs, hidden = self.lstm(embedded)
        outputs = self.fc(outputs.permute(1, 0, 2))[:, :-1, :].permute(0, 2, 1)

        target_tokens = context.t()[:, 1:]
        loss = criterion(input=outputs, target=target_tokens)

        return loss


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_vocab_size,
        output_vocab_size,
        max_length,
        start_id,
        stop_id
    ):
        super(EncoderDecoder, self).__init__()
        # YOUR CODE HERE
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.dropout_p = nn.Dropout(0.1)
        self.max_length = max_length
        self.teacher_forcing_ratio = 0.5
        self.START_id = start_id
        self.STOP_id = stop_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Encoder related
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.encoder_embedding = nn.Embedding(input_vocab_size, hidden_size)

        # Decoder related
        self.decoder_embedding = nn.Embedding(self.output_vocab_size, self.hidden_size)
        self.decoder_attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.decoder_attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.decoder = nn.GRU(self.hidden_size, self.hidden_size)
        self.decoder_out = nn.Linear(self.hidden_size, self.output_vocab_size)
    
    def encoder_forward(self, input, hidden):
        embedded = self.encoder_embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.encoder(output, hidden)
        return output, hidden
    
    def decoder_forward(self, input, hidden, encoder_outputs):
        embedded = self.decoder_embedding(input).view(1, 1, -1)
        embedded = self.dropout_p(embedded)

        attn_weights = nn.functional.softmax(
        self.decoder_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                    encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.decoder_attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.decoder(output, hidden)

        output = nn.functional.log_softmax(self.decoder_out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    
    def forward(self, inputs, criterion, targets=None):
        encoder_hidden = self.initHidden()
        
        input_tensor = inputs[0][0].reshape(self.max_length, 1)
        target_tensor = inputs[1][0].reshape(self.max_length, 1)
        
        #print(input_tensor)
        #print(target_tensor)
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        #print(input_length)
        #print(target_length)
        
        encoder_outputs = torch.zeros(self.max_length, self.hidden_size, device=self.device)
        
        loss = 0
        
        #print("Starting encoder work.")
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder_forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        #print("Completed encoder work.")
        
        decoder_input = torch.tensor([[self.START_id]], device=self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        
        #print("Starting decoder work.")
        if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder_forward(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder_forward(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di])
                #print("Got loss!")
                if decoder_input.item() == self.STOP_id:
                    break
        
        #print("Returning loss")
        return loss
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
