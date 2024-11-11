from maestro.util import tqdm
import torch
import torch.nn as nn
import shutil
from pathlib import Path
import os
from maestro.eval.gtzan import eval_laion_clap
from maestro.constants import source_to_label, label_to_source
from contextlib import nullcontext
import json
import numpy as np

class GTZANLogEntry:
    def __init__(self, epoch, train_loss=None, val_loss=None,
                 accuracy=None, source_accuracies=None,
                 source_train_losses=None, source_val_losses=None, top_k=None):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.accuracy = accuracy
        self.top_k = top_k

        self.source_accuracies = {} if source_accuracies is None else source_accuracies
        self.source_train_losses = {} if source_train_losses is None else source_train_losses
        self.source_val_losses = {} if source_val_losses is None else source_val_losses

    def to_dict(self):
        base_dict = {
            'epoch': self.epoch,
        }

        # Add optional fields
        if self.train_loss is not None:
            base_dict['train_loss'] = self.train_loss
        if self.val_loss is not None:
            base_dict['val_loss'] = self.val_loss
        if self.accuracy is not None:
            # Convert numpy arrays to lists
            if isinstance(self.accuracy, np.ndarray):
                base_dict['accuracy'] = self.accuracy.tolist()
            else:
                base_dict['accuracy'] = self.accuracy
        if self.top_k is not None:
            base_dict['top_k'] = self.top_k
        if len(self.source_accuracies) > 0:
            # Convert numpy arrays in source_accuracies to lists
            base_dict['source_accuracies'] = {
                source: acc.tolist() if isinstance(acc, np.ndarray) else acc
                for source, acc in self.source_accuracies.items()
            }
        if len(self.source_train_losses) > 0:
            base_dict['source_train_losses'] = self.source_train_losses
        if len(self.source_val_losses) > 0:
            base_dict['source_val_losses'] = self.source_val_losses
        
        return base_dict


    def print_train_loss(self):
        #Print total loss
        print(f"Epoch {self.epoch + 1} - Training loss: {self.train_loss:.3f}")

        #Print source losses on one line
        if len(self.source_train_losses) > 0:
            line = "By source:"
            for source, loss in self.source_train_losses.items():
                line += f" {label_to_source(source)}: {loss:.3f},"
            line = line[:-1] #Remove last comma
            print(line)

    def print_val_loss(self):
        #Print total loss
        print(f"Epoch {self.epoch + 1} - Validation loss: {self.val_loss:.3f}")

        #Print source losses on one line
        if len(self.source_val_losses) > 0:
            line = "By source:"
            for source, loss in self.source_val_losses.items():
                line += f" {label_to_source(source)}: {loss:.3f},"
            line = line[:-1]
            print(line)

    def print_accuracy(self):
        if self.accuracy is None or self.top_k is None:
            return
        
        if isinstance(self.top_k, int):
            print(f"Epoch {self.epoch + 1} - Validation accuracy: {self.accuracy:.3f}")

            if len(self.source_accuracies) > 0:
                line = "By source:"
                for source, acc in self.source_accuracies.items():
                    line += f" {label_to_source(source)}: {acc}:.3f,"
                line = line[:-1]
                print(line)
        else:
            # Overall
            msg = f"Epoch {self.epoch + 1} - Validation accuracy:"
            for i, k in enumerate(self.top_k):
                msg += f" @{k}: {self.accuracy[i]:.3f},"
            msg = msg[:-1]
            print(msg)

            # By source
            if len(self.source_accuracies) > 0:
                for source, acc in self.source_accuracies.items():
                    line = f"For source {source}:"
                    for i, k in enumerate(self.top_k):
                        line += f" @{k}: {acc[i]:.3f},"
                    line = line[:-1]
                    print(line)

    

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
        weighted_source_losses = [source_losses[source] * source_weights[source] for source in source_weights.keys()]
        batch_loss = sum(weighted_source_losses)

    return batch_loss, source_losses

#TODO: rename
def _gtzan_cross_entropy_loss_old(criterion, *, label_embeddings, audio_embeddings, labels, temperature):
    logits = torch.matmul(audio_embeddings, label_embeddings.T) / temperature
    return criterion(logits, labels)

def train_gtzan_model(*, output_dir, train_loader, val_loader, model, epochs, device, optimizer, scheduler, label_ids, label_attn_masks,
                      loss_by_source=False ,source_weights=None, temperature = 1.0, disable_tokenizers_parallelism = True, clear_output_dir = False, 
                      ckpt_interval = None, top_k=None, accuracy=False):
    
    #Normalize source weights
    if source_weights is not None:
        source_weights = {source: weight / sum(source_weights.values()) for source, weight in source_weights}

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

    log_dicts = []

    # Move model and constant tensors to device
    model.to(device)
    label_ids = label_ids.to(device)
    label_attn_masks = label_attn_masks.to(device)
    
    # Init loss
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Train loop
    for epoch in range(epochs):

        log_entry = GTZANLogEntry(epoch, top_k=top_k[0] if scalar_k else top_k)

        # ----- Training -----
        model.train()
        train_loss = 0
        source_train_losses = {}
        source_batch_counts = {}

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

                if source not in source_batch_counts:
                    source_batch_counts[source] = 0
                source_batch_counts[source] += 1

            train_loss += batch_loss.item()

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
        
        # Calc loss
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Avg source losses
        for source, loss in source_train_losses.items():
            source_train_losses[source] = loss / source_batch_counts[source]
        
        log_entry.train_loss = avg_train_loss
        log_entry.source_train_losses = source_train_losses

        log_entry.print_train_loss()


        # ----- Validation -----
        model.eval()
        val_loss = 0
        source_val_losses = {}
        source_batch_counts = {}

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

                    if source not in source_batch_counts:
                        source_batch_counts[source] = 0
                    source_batch_counts[source] += 1
            
            # Avg loss
            final_val_loss = val_loss / len(val_loader)
            
            #Avg source losses
            for source, loss in source_val_losses.items():
                source_val_losses[source] = loss / source_batch_counts[source]
            
            log_entry.val_loss = final_val_loss
            log_entry.source_val_losses = source_val_losses

            log_entry.print_val_loss()

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
                        source_loader = torch.utils.data.DataLoader(source_set, batch_size=32, shuffle=False, collate_fn=gtzan_collate)
                        source_accs[source] = eval_laion_clap(model, 
                                                              val_loader = source_loader,
                                                              label_embeddings=label_embeddings,
                                                              top=top_k, desc=f"Evaluating {source} accuracy",
                                                              leave=False)
                    
                    # Calculate weighted average by source presence
                    total_samples = len(val_loader.dataset)
                    total_acc = np.zeros(len(top_k))
                    for source, acc in source_accs.items():
                        
                        acc_np = np.array(acc)
                        source_samples = len(source_indices[source])
                        total_acc += acc_np * source_samples / total_samples
                    acc = total_acc

                    log_entry.accuracy = acc
                    log_entry.source_accuracies = source_accs
                else:
                    acc = eval_laion_clap(model, 
                                          val_loader = val_loader,
                                          label_embeddings=label_embeddings,
                                          top=top_k,
                                          leave=False)
                    log_entry.accuracy = acc
                log_entry.print_accuracy()
        
        # Save checkpoint
        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pt")

        # Save log
        log_dicts.append(log_entry.to_dict())
        
        with open(output_dir / "log.json", 'w') as f:
            json.dump(log_dicts, f)
        
        #TODO: graphs, cm
    
    torch.save(model.state_dict(), output_dir / f"model_final.pt")
