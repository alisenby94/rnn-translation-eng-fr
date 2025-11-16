"""
Evaluate model performance by sentence length.

This script loads a trained model and evaluates its performance on the validation set,
breaking down accuracy and loss by sentence length buckets.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.arch.rnn_model import create_rnn_model
from src.arch.lstm_model import create_lstm_model
from src.arch.gru_model import create_gru_model
from src.tools.data_utils import load_data_from_files


def evaluate_by_length(model, val_loader, device, length_buckets=[(1,5), (6,10), (11,15), (16,20), (21,100)]):
    """
    Evaluate model performance grouped by sentence length.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        length_buckets: List of (min, max) tuples for grouping sentence lengths
        
    Returns:
        Dictionary with results per length bucket
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    # Store metrics per bucket
    bucket_metrics = defaultdict(lambda: {
        'total_loss': 0.0,
        'correct_tokens': 0,
        'total_tokens': 0,
        'num_sentences': 0
    })
    
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            
            # Forward pass
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            
            # Get predictions and targets
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            
            # Calculate loss per token (no reduction)
            loss_per_token = criterion(output_flat, trg_flat)
            
            # Get predictions
            predictions = output_flat.argmax(1)
            correct = (predictions == trg_flat) & (trg_flat != 0)
            
            # Process each sentence in batch
            batch_size = src.size(0)
            seq_len = trg.size(1) - 1  # Exclude first token
            
            loss_per_token = loss_per_token.view(batch_size, seq_len)
            correct = correct.view(batch_size, seq_len)
            trg_tokens = trg[:, 1:]
            
            for i in range(batch_size):
                # Count non-padding tokens (actual sentence length)
                mask = trg_tokens[i] != 0
                sent_length = mask.sum().item()
                
                if sent_length == 0:
                    continue
                
                # Find which bucket this sentence belongs to
                for min_len, max_len in length_buckets:
                    if min_len <= sent_length <= max_len:
                        bucket_key = f"{min_len}-{max_len}"
                        
                        # Accumulate metrics
                        sent_loss = loss_per_token[i][mask].sum().item()
                        sent_correct = correct[i][mask].sum().item()
                        
                        bucket_metrics[bucket_key]['total_loss'] += sent_loss
                        bucket_metrics[bucket_key]['correct_tokens'] += sent_correct
                        bucket_metrics[bucket_key]['total_tokens'] += sent_length
                        bucket_metrics[bucket_key]['num_sentences'] += 1
                        break
    
    # Calculate averages per bucket
    results = {}
    for bucket, metrics in bucket_metrics.items():
        if metrics['total_tokens'] > 0:
            results[bucket] = {
                'avg_loss': metrics['total_loss'] / metrics['total_tokens'],
                'accuracy': metrics['correct_tokens'] / metrics['total_tokens'],
                'num_sentences': metrics['num_sentences'],
                'total_tokens': metrics['total_tokens']
            }
    
    return results


def plot_results(results_dict, save_path='results/performance_by_length.png', min_samples=10):
    """
    Plot performance metrics by sentence length for multiple models.
    
    Args:
        results_dict: Dict of {model_name: results}
        save_path: Path to save the plot
        min_samples: Minimum samples to show warning
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    for model_name, results in results_dict.items():
        # Sort buckets by first number
        sorted_buckets = sorted(results.items(), key=lambda x: int(x[0].split('-')[0]))
        
        buckets = [b[0] for b in sorted_buckets]
        accuracies = [b[1]['accuracy'] * 100 for b in sorted_buckets]
        losses = [b[1]['avg_loss'] for b in sorted_buckets]
        num_sentences = [b[1]['num_sentences'] for b in sorted_buckets]
        
        # Plot accuracy
        ax1.plot(buckets, accuracies, marker='o', label=model_name.upper(), linewidth=2)
        
        # Plot loss
        ax2.plot(buckets, losses, marker='o', label=model_name.upper(), linewidth=2)
        
        # Plot sample counts (just once)
        if model_name == list(results_dict.keys())[0]:
            ax3.bar(buckets, num_sentences, alpha=0.6)
    
    # Accuracy plot
    ax1.set_title('Accuracy by Sentence Length', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sentence Length (tokens)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Loss plot
    ax2.set_title('Loss by Sentence Length', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sentence Length (tokens)')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample count plot
    ax3.set_title('Number of Samples per Length', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sentence Length (tokens)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # Add horizontal line for minimum sample threshold
    ax3.axhline(y=min_samples, color='red', linestyle='--', linewidth=1, 
                label=f'Min samples threshold ({min_samples})', alpha=0.7)
    ax3.legend()
    
    # Annotate counts on bars
    first_model_results = list(results_dict.values())[0]
    sorted_buckets = sorted(first_model_results.items(), key=lambda x: int(x[0].split('-')[0]))
    for i, (bucket, metrics) in enumerate(sorted_buckets):
        count = metrics['num_sentences']
        ax3.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance by sentence length')
    parser.add_argument('--models', nargs='+', required=True, choices=['rnn', 'lstm', 'gru'],
                        help='Models to evaluate (e.g., --models rnn lstm gru)')
    parser.add_argument('--input_file', default='data/small_vocab_en')
    parser.add_argument('--target_file', default='data/small_vocab_fr')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--min_samples', type=int, default=10,
                        help='Minimum samples required per bucket to include in results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data once
    print("Loading data...")
    _, val_loader, prep = load_data_from_files(
        args.input_file, args.target_file, args.batch_size, args.val_split
    )
    
    # Define length buckets
    length_buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 100)]
    
    # Evaluate each model
    all_results = {}
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model...")
        print('='*60)
        
        # Load model
        model_path = f'models/{model_type}_model.pt'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print(f"   Please train the model first: python src/train.py --model {model_type}")
            continue
        
        # Create model architecture
        if model_type == 'rnn':
            model = create_rnn_model(len(prep.src_vocab), len(prep.trg_vocab))
        elif model_type == 'lstm':
            model = create_lstm_model(len(prep.src_vocab), len(prep.trg_vocab))
        else:
            model = create_gru_model(len(prep.src_vocab), len(prep.trg_vocab))
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        print(f"Loaded model from: {model_path}")
        
        # Evaluate by length
        results = evaluate_by_length(model, val_loader, device, length_buckets)
        all_results[model_type] = results
        
        # Print results
        print(f"\nResults for {model_type.upper()}:")
        print(f"{'Length Range':<15} {'Accuracy':<12} {'Avg Loss':<12} {'# Sentences':<12} {'# Tokens':<12}")
        print('-' * 63)
        
        total_sentences = sum(m['num_sentences'] for m in results.values())
        
        for bucket in sorted(results.keys(), key=lambda x: int(x.split('-')[0])):
            metrics = results[bucket]
            num_sent = metrics['num_sentences']
            pct_of_total = (num_sent / total_sentences * 100) if total_sentences > 0 else 0
            
            # Warn if sample size is small
            warning = " ⚠️  LOW SAMPLE" if num_sent < args.min_samples else ""
            
            print(f"{bucket:<15} "
                  f"{metrics['accuracy']*100:>10.2f}%  "
                  f"{metrics['avg_loss']:>10.4f}  "
                  f"{num_sent:>10} ({pct_of_total:>5.1f}%)  "
                  f"{metrics['total_tokens']:>10}"
                  f"{warning}")
        
        print(f"\nTotal validation sentences: {total_sentences}")
        print(f"⚠️  Buckets with < {args.min_samples} samples may not be statistically significant")
    
    # Plot comparison if multiple models
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("Generating comparison plot...")
        plot_results(all_results, min_samples=args.min_samples)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
