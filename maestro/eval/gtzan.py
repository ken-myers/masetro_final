from maestro.util import tqdm
import torch
import numpy as np

def eval_laion_clap(model, *, val_loader, label_embeddings, top=1, confusion_matrix=False):
    
    label_embeddings = label_embeddings.to(model.device)

    correct_labels = []
    rankings = []

    if isinstance(top, list):
        max_k = max(top)
        return_scalar = False
    elif isinstance(top, int):
        max_k = top
        return_scalar = True
    else:
        raise ValueError("top must be an int or a list of ints")


    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating accuracy"):
            labels = batch['label'].to(model.device)
            features = batch['features'].to(model.device)

            audio_embeddings = model.get_audio_features(features)

            logits = torch.matmul(audio_embeddings, label_embeddings.T)

            # Get predictions
            _, preds = torch.topk(logits, max_k, dim=1)

            rankings.append(preds.cpu().numpy())
            correct_labels.extend(labels.cpu().numpy())
        
        correct_labels = np.array(correct_labels)
        rankings = np.vstack(rankings)
    

    total = len(correct_labels)

    if return_scalar:
        r = 0
        for correct, rank in zip(correct_labels, rankings):
            if correct in rank[:top]:
                r += 1
        return r / total
    else:
        r = [0] * len(top)
        for correct, rank in zip(correct_labels, rankings):
            for i, k in enumerate(top):
                if correct in rank[:k]:
                    r[i] += 1
        return [ri / total for ri in r]