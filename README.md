# English to French Translation with RNNs

Implementation and comparison of three vanilla sequence-to-sequence models for machine translation:
- **2-Layer RNN** (Vanilla Recurrent Neural Network)
- **2-Layer LSTM** (Long Short-Term Memory)
- **2-Layer GRU** (Gated Recurrent Unit)

## Architecture

Vanilla encoder-decoder sequence-to-sequence models implemented in PyTorch:

**Encoder (2-layer stacked):**
- Word embeddings → 2-layer RNN/LSTM/GRU → final hidden state

**Decoder (2-layer stacked):**
- Takes encoder's final state as context → generates translation word-by-word

**Key characteristics:**
- No attention mechanism (vanilla architecture)
- Teacher forcing during training (use ground truth output as input instead of the model's previous prediction)
- Dropout regularization
- Greedy decoding for inference

## Setup

```bash
git clone https://github.com/alisenby94/rnn-translation-eng-fr.git
cd rnn-translation-eng-fr
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.4+, NumPy 2.0+

**Dataset:** 137,860 parallel English-French sentence pairs (small vocabulary corpus) in `data/`

## Quick Start

**Train a model:**
```bash
python src/train.py --model rnn --input_file data/small_vocab_en --target_file data/small_vocab_fr --epochs 10
```

**Evaluate by sentence length:**
```bash
python src/evaluate_by_length.py --models rnn lstm gru
```

**Training options:**
- `--model`: Model type (`rnn`, `lstm`, `gru`)
- `--epochs`: Training epochs (default: 10)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)

## Results

Models are saved to `models/` with training history in JSON format. Training curves saved as PNG in `results/`.

**Vocabularies are cached** to `data/` after first run (207 English words, 359 French words).

## Model Comparison

| Model | Parameters | Memory | Long Sequences |
|-------|-----------|---------|----------------|
| RNN | 2.17M | Single hidden state | Poor |
| LSTM | 7.69M | Hidden + cell state | Good |
| GRU | 5.85M | Hidden state + gates | Better |

**Performance by length analysis** available via `evaluate_by_length.py` - shows accuracy/loss breakdown for different sentence lengths (1-5, 6-10, 11-15, 16-20, 21+ tokens).

## Notes

- Vocabularies are automatically cached after first data load
- Custom padding implementation (assignment requirement)
- Teacher forcing ratio: 0.5 during training, 0 during validation
- Train/val split: 80/20 (deterministic - last 20% of dataset is validation)

## Author

**Andrew Lisenby**
- GitHub: [@alisenby94](https://github.com/alisenby94)
- LinkedIn: [andrew-lisenby](https://www.linkedin.com/in/andrew-lisenby/)
