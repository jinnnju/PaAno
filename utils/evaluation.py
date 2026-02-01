import torch
import torch.nn.functional as F
import numpy as np

# Distance-based anomaly scoring 
@torch.inference_mode()
def calculate_anomaly_scores(model, data_loader, memory_bank, device, top_k=3):
    
    model.eval()
    all_scores = []
    memory_bank = F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)


    for data, _ in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        feats = model.embedding(data)  # (B, D)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        feats = F.normalize(feats, dim=1, eps=1e-12)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Cosine similarity & distance
        sims = feats @ memory_bank.T                    # (B, M)
        sims = torch.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        topk_sim, _ = torch.topk(sims, k=top_k, dim=1, largest=True)
        dists = 1.0 - topk_sim
        scores = dists.mean(dim=1)

        scores = torch.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)
        all_scores.extend(scores.cpu().tolist())

    return all_scores


# Patch-to-point score distribution 
def distribute_patch_scores_to_points(patch_scores, patch_size, num_points): 

    patch_scores = np.nan_to_num(np.asarray(patch_scores, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)

    kernel = np.ones(patch_size, dtype=np.float32)
    sums   = np.convolve(patch_scores, kernel, mode='full')[:num_points]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode='full')[:num_points]

    point_scores = np.divide(
        sums, counts,
        out=np.zeros(num_points, dtype=np.float32),
        where=counts != 0
    )
    return np.nan_to_num(point_scores, nan=0.0, posinf=0.0, neginf=0.0)

