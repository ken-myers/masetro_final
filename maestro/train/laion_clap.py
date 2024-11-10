from maestro.util import tqdm
import torch
import torch.nn as nn
import shutil
from pathlib import Path
import os
from maestro.eval.gtzan import eval_laion_clap

def gtzan_collate(batch):
    return {
        'label': torch.tensor([item['label'] for item in batch]),
        'features': torch.stack([item['features'] for item in batch])
    }

def train_gtzan_model(*, output_dir, train_loader, val_loader, model, epochs, device, optimizer, scheduler, label_ids, label_attn_masks,
                      disable_tokenizers_parallelism = True, clear_output_dir = False, ckpt_interval = None, top_k=None):

    
    if top_k is not None:
        scalar_k = not isinstance(top_k, list)
        if not scalar_k:
            top_k = sorted(top_k)

    if disable_tokenizers_parallelism:
        print("Disabling tokenizers parallelism")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    output_dir = Path(output_dir)
    
    if clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    label_ids = label_ids.to(device)
    label_attn_masks = label_attn_masks.to(device)
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Move data to device
            labels = batch['label'].to(device)
            features = batch['features'].to(device)

            # Generate label embeddings
            label_embeddings = model.get_text_features(label_ids, attention_mask=label_attn_masks)

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
        
        # Calc loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss}")

        # ----- Validation -----
        model.eval()
        val_loss = 0
        
        # Only need to compute label embeddings once since we're not training
        label_embeddings = model.get_text_features(label_ids, attention_mask=label_attn_masks)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                
                # Move data to device
                labels = batch['label'].to(device)
                features = batch['features'].to(device)

                # Generate embeddings
                audio_embeddings = model.get_audio_features(features)

                # Sim matrix
                logits = torch.matmul(audio_embeddings, label_embeddings.T)

                # Compute loss
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        #Calc val loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} - Validation loss: {avg_val_loss}")

        # Calc accuracy
        if top_k is not None:
            accuracy = eval_laion_clap(model, val_loader=val_loader, label_embeddings=label_embeddings, top=top_k)

            if scalar_k:
                print(f"Epoch {epoch + 1} - Accuracy: {accuracy}")
            else:
                acc_msg = f"Epoch {epoch+1} - Accuracy:"
                for k, acc in zip(top_k, accuracy):
                    acc_msg += f" Top {k}: {acc}"
                print(acc_msg)

        # Step the scheduler
        scheduler.step(val_loss)

        # Save checkpoint
        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pt")
        
        #TODO: graphs, accuracy
    
    torch.save(model.state_dict(), output_dir / f"model_final.pt")
