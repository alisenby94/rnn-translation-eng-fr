import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from collections import Counter


class Vocabulary:
    """Simple word/index mapping."""
    
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.word_counts = Counter()
    
    def add_sentence(self, sentence):
        """Count words in a sentence."""
        for word in sentence.split():
            self.word_counts[word] += 1
    
    def build(self, max_words=10000):
        """Build vocab from most common words."""
        for word, _ in self.word_counts.most_common(max_words):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, sentence):
        """Sentence -> indices."""
        unk = self.word2idx['<unk>']
        return [self.word2idx.get(word, unk) for word in sentence.split()]
    
    def decode(self, indices):
        """Indices -> sentence."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<unk>')
            if word in ['<pad>', '<start>']:
                continue
            if word == '<end>':
                break
            words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    """Just holds sentences and vocab."""
    
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        """Get one sentence pair."""
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]
        
        # Encode
        src_ids = self.src_vocab.encode(src)
        trg_ids = self.trg_vocab.encode(trg)
        
        # Add <start> and <end> to target
        trg_ids = [1] + trg_ids + [2]  # 1=<start>, 2=<end>
        
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)


def pad(sequences, padding_value=0):
    """Pad sequences to same length."""
    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)
    
    padded = torch.full((batch_size, max_len), padding_value, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    return padded


def collate_fn(batch):
    """Pad a batch of sequences."""
    src_batch, trg_batch = zip(*batch)
    return pad(src_batch), pad(trg_batch)

# Dataset was already collated properly. Probably good practice for redundancy.
def clean_sentence(sentence):
    """Lowercase and add spaces around punctuation."""
    sentence = sentence.lower()
    for char in [',', '.', '!', '?', "'"]:
        sentence = sentence.replace(char, f' {char}')
    return ' '.join(sentence.split())


def load_data_from_files(input_file, target_file, batch_size=16, val_split=0.2, vocab_cache_dir='data'):
    """Load data, build vocabs, return dataloaders."""
    
    # Read files and clean sentences
    with open(input_file, 'r') as f:
        src_lines = [clean_sentence(line.strip()) for line in f if line.strip()]
    
    with open(target_file, 'r') as f:
        trg_lines = [clean_sentence(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(src_lines)} sentences")
    
    # Check for cached vocabularies
    os.makedirs(vocab_cache_dir, exist_ok=True)
    src_vocab_path = f'{vocab_cache_dir}/src_vocab.pkl'
    trg_vocab_path = f'{vocab_cache_dir}/trg_vocab.pkl'
    
    if os.path.exists(src_vocab_path) and os.path.exists(trg_vocab_path):
        print(f"Loading cached vocabularies from {vocab_cache_dir}/...")
        with open(src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(trg_vocab_path, 'rb') as f:
            trg_vocab = pickle.load(f)
    else:
        print("Building vocabularies...")
        src_vocab = Vocabulary()
        trg_vocab = Vocabulary()
        
        for sent in src_lines:
            src_vocab.add_sentence(sent)
        for sent in trg_lines:
            trg_vocab.add_sentence(sent)
        
        src_vocab.build(max_words=10000)
        trg_vocab.build(max_words=10000)
        
        # Save vocabs
        with open(src_vocab_path, 'wb') as f:
            pickle.dump(src_vocab, f)
        with open(trg_vocab_path, 'wb') as f:
            pickle.dump(trg_vocab, f)
        print(f"Vocabularies saved to {vocab_cache_dir}/")
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")
    
    # Split train/val
    split_idx = int(len(src_lines) * (1 - val_split))
    
    train_dataset = TranslationDataset(
        src_lines[:split_idx],
        trg_lines[:split_idx],
        src_vocab,
        trg_vocab
    )
    
    val_dataset = TranslationDataset(
        src_lines[split_idx:],
        trg_lines[split_idx:],
        src_vocab,
        trg_vocab
    )
    
    print(f"Splitting data: {len(train_dataset)} training, {len(val_dataset)} validation\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Return vocabs in a simple object
    class SimplePreprocessor:
        def __init__(self, src_vocab, trg_vocab):
            self.src_vocab = src_vocab
            self.trg_vocab = trg_vocab
    
    return train_loader, val_loader, SimplePreprocessor(src_vocab, trg_vocab)
