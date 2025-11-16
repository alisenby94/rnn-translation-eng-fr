import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        self.hidden_units = hidden_units
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Dropout (Not strictly necessary for vanilla RNNs)
        self.dropout = nn.Dropout(dropout)
        
        # Stacked 2-layer RNN
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
    
    def forward(self, x):
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(x))
        
        # RNN forward pass
        encoder_outputs, hidden = self.rnn(embedded)
        
        return encoder_outputs, hidden


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, dropout=0.1):
        super(RNNDecoder, self).__init__()
        self.hidden_units = hidden_units
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stacked 2-layer RNN
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_units, vocab_size)
    
    def forward(self, x, hidden):
        # Embedding: (batch_size, 1) -> (batch_size, 1, embedding_dim)
        embedded = self.dropout(self.embedding(x))
        
        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)
        
        # Output: (batch_size, 1, hidden_units) -> (batch_size, hidden_units)
        output = output.squeeze(1)
        
        # Predictions: (batch_size, vocab_size)
        predictions = self.fc(output)
        
        return predictions, hidden


class RNNSeq2Seq(nn.Module):
    def __init__(self, 
                 input_vocab_size, 
                 target_vocab_size,
                 embedding_dim=256,
                 hidden_units=512,
                 dropout=0.1):
        super(RNNSeq2Seq, self).__init__()
        self.hidden_units = hidden_units
        
        self.encoder = RNNEncoder(
            input_vocab_size, 
            embedding_dim, 
            hidden_units, 
            dropout
        )
        
        self.decoder = RNNDecoder(
            target_vocab_size, 
            embedding_dim, 
            hidden_units, 
            dropout
        )
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        # Encode
        _, hidden = self.encoder(src)
        
        # First input to decoder is the <start> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode step by step
        for t in range(1, trg_len):
            # Forward through decoder
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Next input is current target if teacher forcing, else predicted token
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs


def create_rnn_model(input_vocab_size, target_vocab_size, embedding_dim=256, 
                     hidden_units=512, dropout=0.1):
    model = RNNSeq2Seq(
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout=dropout
    )
    
    return model
