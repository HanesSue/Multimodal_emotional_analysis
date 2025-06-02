# voting.py
import torch
import torch.nn.functional as F

def vote_majority(predictions):
    # predictions: List of frame-level logits [T, num_classes]
    pred_labels = [torch.argmax(p) for p in predictions]
    final_label = torch.mode(torch.tensor(pred_labels))[0].item()
    return final_label

def vote_average(predictions):
    probs = [F.softmax(p, dim=0) for p in predictions]
    avg_prob = torch.stack(probs).mean(dim=0)
    return torch.argmax(avg_prob).item()

def vote_weighted(predictions, weights):
    # weights: importance for each frame (length T)
    probs = [F.softmax(p, dim=0) for p in predictions]
    weighted = torch.stack([w*p for w, p in zip(weights, probs)])
    final_prob = weighted.sum(dim=0) / sum(weights)
    return torch.argmax(final_prob).item()
