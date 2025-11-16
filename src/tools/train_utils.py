import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=0.001):
    """All 3 models share the same training procedure."""

    # Move model to gpu if available
    model = model.to(device)

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Main training loop
    for epoch in range(epochs):
        model.train()

        # history
        train_loss, train_acc, train_tokens = 0, 0, 0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        # for each batch move sample sources and targets to device and train
        for src, trg in train_pbar:
            # move batch to device
            src, trg = src.to(device), trg.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)
            output_dim = output.shape[-1]
            
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += ((output_flat.argmax(1) == trg_flat) & (trg_flat != 0)).sum().item()
            train_tokens += (trg_flat != 0).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss, val_acc, val_tokens = 0, 0, 0
        
        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for src, trg in val_pbar:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                
                output_flat = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                
                loss = criterion(output_flat, trg_flat)
                val_loss += loss.item()
                val_acc += ((output_flat.argmax(1) == trg_flat) & (trg_flat != 0)).sum().item()
                val_tokens += (trg_flat != 0).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc / train_tokens if train_tokens > 0 else 0)
        history['val_acc'].append(val_acc / val_tokens if val_tokens > 0 else 0)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"loss: {history['train_loss'][-1]:.4f} - "
              f"accuracy: {history['train_acc'][-1]:.4f} - "
              f"val_loss: {history['val_loss'][-1]:.4f} - "
              f"val_accuracy: {history['val_acc'][-1]:.4f}")
    
    return model, history


def plot_history(history, save_path='results/training_history.png'):
    """Plot training history."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plots saved to {save_path}")


def save_model_and_history(model, history, model_path, history_path):
    """Save model and training history."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
