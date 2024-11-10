from maestro.util import tqdm
import torch
import torch.nn as nn
import shutil
from pathlib import Path
import os

def gtzan_collate(batch):
    return {
        'label': torch.tensor([item['label'] for item in batch]),
        'features': torch.stack([item['features'] for item in batch])
    }

def train_gtzan_model(*, output_dir, train_loader, val_loader, model, epochs, device, optimizer, scheduler, label_ids,
                      disable_tokenizers_parallelism = True, clear_output_dir = False, ckpt_interval = None):

    if disable_tokenizers_parallelism:
        print("Disabling tokenizers parallelism")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    output_dir = Path(output_dir)
    
    if clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    label_ids = label_ids.to(device)
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Move data to device
            labels = batch['label'].to(device)
            features = batch['features'].to(device)

            # Generate label embeddings
            label_embeddings = model.get_text_features(label_ids)

            # Generate audio embeddings
            audio_embeddings = model.get_audio_features(features)
            
            # Compute similarity between label embeddings and audio embeddings
            logits = torch.matmul(audio_embeddings, label_embeddings.T)

            # Compute loss
            loss = criterion(logits, labels)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} - Training loss: {train_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                
                # Move data to device
                labels = batch['label'].to(device)
                features = batch['features'].to(device)

                # Generate embeddings
                label_embeddings = model.get_text_features(label_ids)
                audio_embeddings = model.get_audio(features)

                # Sim matrix
                logits = torch.matmul(audio_embeddings, label_embeddings.T)

                # Compute loss
                loss = criterion(logits, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch} - Validation loss: {val_loss / len(val_loader)}")

        val_loss = val_loss / len(val_loader)
        train_loss = train_loss / len(train_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pt")

        #TODO: graphs, accuracy
    
    torch.save(model.state_dict(), output_dir / f"model_final.pt")
