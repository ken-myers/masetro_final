from maestro.util import tqdm
import torch
import torch.nn as nn
import shutil
from pathlib import Path
import os
from maestro.eval.gtzan import eval_laion_clap
from maestro.constants import source_to_label, label_to_source
from contextlib import nullcontext
import numpy as np
from collections import Counter
from maestro.train.laion_clap.logging import GTZANLogger

def gtzan_collate(batch):
    return {
        'label': torch.tensor([item['label'] for item in batch]),
        'features': torch.stack([item['features'] for item in batch]),
        'sources': torch.tensor([source_to_label(item['source']) for item in batch], dtype=torch.uint8)
    }


def _gtzan_cross_entropy_loss(criterion, *, label_embeddings, audio_embeddings, labels, temperature, sources, source_weights=None, loss_by_source=False):
    source_losses = {}

    sources_present = set(sources.cpu().numpy().tolist())

    # Compute loss
    if loss_by_source or source_weights is not None:
        with torch.no_grad() if source_weights is None else nullcontext():
            for source in sources_present:
                source_mask = sources == source
                if source_mask.sum() == 0:
                    source_losses[source] = torch.tensor(0.0, device=label_embeddings.device)
                else:
                    source_loss = _gtzan_cross_entropy_loss_old(criterion,
                                                    label_embeddings = label_embeddings,
                                                    audio_embeddings=audio_embeddings[source_mask],
                                                    labels=labels[source_mask],
                                                    temperature=temperature,
                                                    )
                    source_losses[source] = source_loss
    if source_weights is None:
        batch_loss = _gtzan_cross_entropy_loss_old(criterion,
                                        label_embeddings = label_embeddings,
                                        audio_embeddings=audio_embeddings,
                                        labels=labels,
                                        temperature=temperature)
    else:
        weighted_source_losses = [source_losses[source] * source_weights[source] for source in sources_present]
        batch_loss = sum(weighted_source_losses)

    return batch_loss, source_losses

#TODO: rename
def _gtzan_cross_entropy_loss_old(criterion, *, label_embeddings, audio_embeddings, labels, temperature):
    logits = torch.matmul(audio_embeddings, label_embeddings.T) / temperature
    return criterion(logits, labels)

def train_gtzan_model(*, output_dir, train_loader, val_loader, model, epochs, device, optimizer, scheduler, label_ids, label_attn_masks,
                      loss_by_source=False ,source_weights=None, temperature = 1.0, disable_tokenizers_parallelism = True, clear_output_dir = False, 
                      ckpt_interval = None, top_k=None, save_graphs=False, save_cms=False, graph_title="GTZAN Model"):
    
    # Init logger
    logger = GTZANLogger(top_k=top_k, loss_by_source=loss_by_source)

    #Normalize source weights
    if source_weights is not None:
        source_weights = {source: weight / sum(source_weights.values()) for source, weight in source_weights.items()}

    # Handle multi-val top k
    if top_k is not None:
        scalar_k = not isinstance(top_k, list)
        if scalar_k:
            top_k = [top_k]
        else:
            top_k = sorted(top_k)
        

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
    criterion = nn.CrossEntropyLoss(reduction='sum')

    if loss_by_source:
        # Count occurence of each source in tain train and val
        train_sources = [item['source'] for item in train_loader.dataset]
        val_sources = [item['source'] for item in val_loader.dataset]

        train_source_counts = Counter(train_sources)
        val_source_counts = Counter(val_sources)




    # Train loop
    for epoch in range(epochs):

        # ----- Training -----
        model.train()
        train_loss = 0
        source_train_losses = {}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Move data to device
            labels = batch['label'].to(device)
            features = batch['features'].to(device)
            sources = batch['sources'].to(device)
            # Generate label embeddings
            label_embeddings = model.get_text_features(label_ids, attention_mask=label_attn_masks)

            # Generate audio embeddings
            audio_embeddings = model.get_audio_features(features)
            
            # Compute loss
            batch_loss, source_losses = _gtzan_cross_entropy_loss(criterion,
                                                                label_embeddings = label_embeddings,
                                                                audio_embeddings=audio_embeddings,
                                                                labels=labels,
                                                                temperature=temperature,
                                                                sources=sources,
                                                                source_weights=source_weights,
                                                                loss_by_source=loss_by_source)
            for source, loss in source_losses.items():
                if source not in source_train_losses:
                    source_train_losses[source] = 0
                source_train_losses[source] += loss.item()

            train_loss += batch_loss.item()

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
        
        # Calc loss
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Avg source losses
        for label, loss in source_train_losses.items():
            source = label_to_source(label)
            source_train_losses[label] = loss / train_source_counts[source]
        
        logger.train_loss(avg_train_loss, source_train_losses)

        # ----- Validation -----
        model.eval()
        val_loss = 0
        source_val_losses = {}

        # Only need to compute label embeddings once since we're not training
        label_embeddings = model.get_text_features(label_ids, attention_mask=label_attn_masks)
        

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", leave=False):
                labels = batch['label'].to(device)
                features = batch['features'].to(device)
                sources = batch['sources'].to(device)

                # Generate audio embeddings
                audio_embeddings = model.get_audio_features(features)

                # Compute loss
                batch_loss, source_losses = _gtzan_cross_entropy_loss(criterion,
                                                                label_embeddings = label_embeddings,
                                                                audio_embeddings=audio_embeddings,
                                                                labels=labels,
                                                                temperature=temperature,
                                                                sources=sources,
                                                                source_weights=source_weights,
                                                                loss_by_source=loss_by_source)
                val_loss += batch_loss.item()

                # Add source losses
                for source, loss in source_losses.items():
                    if source not in source_val_losses:
                        source_val_losses[source] = 0
                    source_val_losses[source] += loss.item()

            
            # Avg loss
            final_val_loss = val_loss / len(val_loader.dataset)
            
            #Avg source losses
            for label, loss in source_val_losses.items():
                source = label_to_source(label)
                source_val_losses[label] = loss / val_source_counts[source]
            
            logger.val_loss(final_val_loss, source_val_losses)

            # Step the scheduler
            scheduler.step(final_val_loss)

            # Calc accuracy
            if top_k is not None:
                if loss_by_source:
                    # Get the dataset from the loader
                    val_set = val_loader.dataset

                    # Get indices for each source
                    source_indices ={}
                    for i, item in enumerate(val_set):
                        source = item['source']
                        if source not in source_indices:
                            source_indices[source] = []
                        source_indices[source].append(i)
                    
                    # Create subset datasets
                    source_sets = {source: torch.utils.data.Subset(val_set, indices) for source, indices in source_indices.items()}
            
                    source_accs = {}
                    for source, source_set in source_sets.items():
                        source_loader = torch.utils.data.DataLoader(source_set, batch_size=val_loader.batch_size, shuffle=False, collate_fn=gtzan_collate)
                        source_accs[source] = eval_laion_clap(model, 
                                                              val_loader = source_loader,
                                                              label_embeddings=label_embeddings,
                                                              top=top_k, desc=f"Evaluating {source} accuracy",
                                                              leave=False, 
                                                            )
                    
                    # Calculate weighted average by source presence
                    total_samples = len(val_loader.dataset)
                    total_acc = np.zeros(len(top_k))
                    for source, acc in source_accs.items():
                        
                        acc_np = np.array(acc)
                        source_samples = len(source_indices[source])
                        total_acc += acc_np * source_samples / total_samples
                    acc = total_acc

                    logger.accuracy(acc, source_accs)
                else:
                    acc = eval_laion_clap(model, 
                                          val_loader = val_loader,
                                          label_embeddings=label_embeddings,
                                          top=top_k,
                                          leave=False)
                    logger.accuracy(acc)
        
        # Save checkpoint
        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pt")

        
        logger.step()

        # Dump logs
        logger.dump_log(output_dir / "log.json")
        
        if save_graphs:
            logger.dump_graphs(output_dir, graph_title)

        # TODO: cm

    torch.save(model.state_dict(), output_dir / f"model_final.pt")
