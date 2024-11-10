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

def _gtzan_cross_entropy_loss(criterion, *, label_embeddings, audio_embeddings, labels, temperature ):
    logits = torch.matmul(audio_embeddings, label_embeddings.T) / temperature
    return criterion(logits, labels)

def train_gtzan_model(*, output_dir, train_loader, val_loader, model, epochs, device, optimizer, scheduler, label_ids, label_attn_masks,
                      val_weights = None, val_loader_names=None, temperature = 1.0, disable_tokenizers_parallelism = True, clear_output_dir = False, ckpt_interval = None, top_k=None):

    # Set flags for multi-val inputs
    if isinstance(val_loader, list):
        multi_val = True
        val_loaders = val_loader
    else:
        multi_val = False
        val_loaders = [val_loader]
        
    if top_k is not None:
        scalar_k = not isinstance(top_k, list)
        if not scalar_k:
            top_k = sorted(top_k)

    # Arg validation
    if multi_val:
        if val_weights is not None and len(val_weights) != len(val_loaders):
            raise ValueError("val_weights must have the same length as val_loader")
        if val_loader_names is not None and len(val_loader_names) != len(val_loaders):
            raise ValueError("val_loader_names must have the same length as val_loader")
    else:
        if val_weights is not None:
            print("Warning: val_weights will be ignored since val_loader is not a list")
        if val_loader_names is not None:
            print("Warning: val_loader_names will be ignored since val_loader is not a list")
    

    # Disable tokenizers parallelism
    if disable_tokenizers_parallelism:
        print("Disabling tokenizers parallelism")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Create output directory
    output_dir = Path(output_dir)

    if clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move model and constant tensors to device
    model.to(device)
    label_ids = label_ids.to(device)
    label_attn_masks = label_attn_masks.to(device)
    
    # Init loss
    criterion = nn.CrossEntropyLoss()
    train_loss_log = []
    val_loss_log = []

    # Train loop
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
            
            # Compute loss
            loss = _gtzan_cross_entropy_loss(criterion,
                                            label_embeddings = label_embeddings,
                                            audio_embeddings=audio_embeddings,
                                            labels=labels,
                                            temperature=temperature)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        # Calc loss
        avg_train_loss = train_loss / len(train_loader)
        train_loss_log.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss}")

        # ----- Validation -----
        model.eval()
        val_losses = [0] * len(val_loaders)
        batch_counts = [0] * len(val_loaders)
        
        # Only need to compute label embeddings once since we're not training
        label_embeddings = model.get_text_features(label_ids, attention_mask=label_attn_masks)
        

        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):
                # Create tag
                if multi_val:
                    if val_loader_names is not None:
                        tag = f" ({val_loader_names[i]})"
                    else:
                        tag = f" ({i})"
                else:
                    tag = ''

                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation" + tag):
                    
                    # Move data to device
                    labels = batch['label'].to(device)
                    features = batch['features'].to(device)

                    # Generate embeddings
                    audio_embeddings = model.get_audio_features(features)

                    loss = _gtzan_cross_entropy_loss(criterion,
                                                    label_embeddings = label_embeddings,
                                                    audio_embeddings=audio_embeddings,
                                                    labels=labels,
                                                    temperature=temperature)
                    val_losses[i] += loss.item()
                    batch_counts[i] += 1
                
                # Calc accuracy
                if top_k is not None:
                    accuracy = eval_laion_clap(model, val_loader=val_loader, label_embeddings=label_embeddings, top=top_k)

                    if scalar_k:
                        print(f"Epoch {epoch + 1} - Accuracy{tag}: {accuracy}")
                    else:
                        acc_msg = f"Epoch {epoch+1} - Accuracy{tag}:"
                        for k, acc in zip(top_k, accuracy):
                            acc_msg += f" Top {k}: {acc}"
                        print(acc_msg)

            
        # Calc final val loss
        if val_weights is not None:
            avg_val_losses = [val_loss / batch_count for val_loss, batch_count in zip(val_losses, batch_counts)]
            final_val_loss = sum([w * l for w, l in zip(val_weights, avg_val_losses)])
        else:
            final_val_loss = sum(avg_val_losses) / sum(batch_counts)

        val_loss_log.append(final_val_loss)

        if multi_val:
            print(f"Epoch {epoch + 1} - {'Weighted' if val_weights is not None else 'Average'} val loss: {final_val_loss}")    

        # Step the scheduler
        scheduler.step(final_val_loss)

        # Save checkpoint
        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pt")
        
        #TODO: graphs, cm
    
    torch.save(model.state_dict(), output_dir / f"model_final.pt")
