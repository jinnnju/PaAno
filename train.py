import copy, math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def train_model(model, train_loader, train_patches, device, num_iter=200, pretext_step=64,
                lr=1e-4, see_loss=None):

    # fixed hyperparams in PaAno
    radius = 2
    lambda_weight = 1
    temperature = 1.0
    num_rand_patches = 5
    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    pos_weight = torch.tensor([1.0]).to(device)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    iteration_count = 0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    print("    [Training Info]")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)

    _offsets = torch.tensor([*range(-radius, 0), *range(1, radius + 1)], dtype=torch.long)

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            iteration_count += 1

            # Update LR
            lr = cosine_annealed_lr(iteration_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze()  # (M,)
            anchors = batch_data
            M = batch_data.shape[0]
            total_len = len(train_patches)

            # positives 
            _cand = batch_indexes.view(-1, 1) + _offsets.view(1, -1)      # (M, 2r)
            _valid = (_cand >= 0) & (_cand < total_len)
            _noise = torch.rand_like(_cand.float())
            _score = torch.where(_valid, _noise, torch.full_like(_noise, -1.0))
            _choice = _score.argmax(dim=1)                                # (M,)
            _pos_idx = _cand.gather(1, _choice.view(-1, 1)).squeeze(1)    # (M,)
            _none_valid = _valid.sum(dim=1) == 0
            if _none_valid.any():
                _pos_idx[_none_valid] = batch_indexes[_none_valid]
            positives = torch.stack([train_patches[i] for i in _pos_idx.tolist()], dim=0).to(device, non_blocking=True)

            if iteration_count < (num_iter / 10) :
                current_lambda_pretext = lambda_weight * (1 - (iteration_count / (num_iter / 10)))
            else:
                current_lambda_pretext = 0.0

            if current_lambda_pretext > 0.0:
                # pretext_patches 
                pretext_patches = []
                pretext_valid_mask = []

                _tgt = batch_indexes - pretext_step
                _pre_mask = (_tgt >= 0) & (_tgt < total_len)
                _tgt_clamped = _tgt.clamp(0, total_len - 1)

                for i in range(M):
                    if _pre_mask[i]:
                        pretext_patches.append(train_patches[_tgt_clamped[i].item()].unsqueeze(0))
                        pretext_valid_mask.append(True)
                    else:
                        pretext_patches.append(torch.zeros_like(train_patches[0].unsqueeze(0)))
                        pretext_valid_mask.append(False)

                pretext_patches = torch.cat(pretext_patches, dim=0).to(device, non_blocking=True)
                pretext_valid_mask = torch.tensor(pretext_valid_mask, dtype=torch.bool, device=device)

                # anchors + positives + pretext
                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos     = all_embeddings[M:2*M]
                h_pretext = all_embeddings[2*M:3*M]

            else:
                pretext_patches    = None
                pretext_valid_mask = None

                # anchors + positives
                all_patches = torch.cat([anchors, positives], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos     = all_embeddings[M:2*M]

            # triplet
            z_anchor = model.projection(h_anchors)
            z_pos    = model.projection(h_pos)

            z_anchor = F.normalize(z_anchor, dim=1)
            z_pos    = F.normalize(z_pos, dim=1)

            _sim_ap  = (z_anchor @ z_pos.T) / temperature         # (M, M)
            pos_sims = _sim_ap.diag()                             # (M,)

            _sim_ap_f = _sim_ap.clone()
            _sim_ap_f.diagonal().fill_(-float('inf'))
            neg_dists = 1 - _sim_ap_f
            hard_neg_dists, _ = torch.max(neg_dists, dim=1)

            pos_dists = 1 - pos_sims
            triplet_loss = F.relu(pos_dists - hard_neg_dists + 0.5).mean()

            # Pretext Task 
            if current_lambda_pretext > 0.0:
                h_pre = h_pretext[pretext_valid_mask]
                h_anchor_pre = h_anchors[pretext_valid_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)

                all_indices = torch.arange(M, device=device)
                anchor_indices = all_indices.repeat_interleave(num_rand_patches)
                rand_offsets = torch.randint(1, M, (M * num_rand_patches,), device=device)
                unadj_indices = (anchor_indices + rand_offsets) % M

                h_unadj = h_anchors[unadj_indices]
                h_anchor_unadj = h_anchors.repeat_interleave(num_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)

                all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                all_pretext_labels = torch.cat([
                    torch.ones(h_concat_pre.size(0), device=device),
                    torch.zeros(h_concat_unadj.size(0), device=device)
                ])

                pretext_outputs = model.classification_head(all_pretext_features).squeeze(1)
                pretext_loss_all = criterion_pretext(pretext_outputs, all_pretext_labels)

                loss_pre = pretext_loss_all[:h_concat_pre.size(0)].mean()
                loss_unadj = pretext_loss_all[h_concat_pre.size(0):].mean()
                pretext_loss = loss_pre + loss_unadj
            else:
                pretext_loss = torch.tensor(0.0, device=device)

            final_loss = triplet_loss + current_lambda_pretext * pretext_loss

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()

            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())

    pbar.close()
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_trained_encoder.pth')
