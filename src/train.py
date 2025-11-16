#!/usr/bin/env python3

# General imports
import os, sys, torch

# Set up path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import model architectures
from src.arch.rnn_model import create_rnn_model
from src.arch.lstm_model import create_lstm_model
from src.arch.gru_model import create_gru_model

# Import data loading and training utilities
from src.tools.data_utils import load_data_from_files
from src.tools.train_utils import train_model, plot_history, save_model_and_history
from src.tools.arch_utils import count_parameters
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--target_file', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    train_loader, val_loader, prep = load_data_from_files(
        args.input_file, args.target_file, args.batch_size, args.val_split)
    
    # Create model
    model_type = args.model.upper()
    print(f"Creating {model_type}...")
    
    if args.model == 'rnn':
        model = create_rnn_model(len(prep.src_vocab), len(prep.trg_vocab))
    elif args.model == 'lstm':
        model = create_lstm_model(len(prep.src_vocab), len(prep.trg_vocab))
    elif args.model == 'gru':
        model = create_gru_model(len(prep.src_vocab), len(prep.trg_vocab))
    else:
        # This should be unreachable due to argparse choices
        raise ValueError(f"Unknown model type: {args.model}\nPlease choose from 'rnn', 'lstm', or 'gru'.")
    
    # print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"Params: {count_parameters(model):,}\n")

    # Train
    print("Training...\n")
    model, hist = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate)
    
    # Save
    save_model_and_history(model, hist, f'models/{args.model}_model.pt', f'models/{args.model}_history.json')
    plot_history(hist, f'results/{args.model}_training_history.png')


if __name__ == "__main__":
    main()
