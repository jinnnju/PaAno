from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import torch.nn.functional as F


def load_model(model_path, device):
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


@torch.no_grad()
def create_memory_bank(model, data_loader, device, num_cores=None):  
    model.eval()
    embeddings = []
    indices = []
    
    for data, batch_indices in data_loader:
        data = data.to(device)
        h = model.embedding(data)
        embeddings.append(h.detach().cpu().float())  
        indices.append(batch_indices.detach().cpu())

    embeddings_tensor = torch.cat(embeddings, dim=0)
    indices_tensor    = torch.cat(indices, dim=0)
    num_samples       = embeddings_tensor.size(0)

 
    if num_cores is None:
        return embeddings_tensor, indices_tensor

  
    if isinstance(num_cores, float):
        k = int(round(num_cores * num_samples))    
    else:
        k = int(num_cores)

    min_cores_eff = min(500, max(1, num_samples - 1)) 
    num_cores = max(min_cores_eff, min(k, num_samples - 1))

    if num_cores >= num_samples:
        return embeddings_tensor, indices_tensor

  
    flattened = embeddings_tensor.view(num_samples, -1)
    flattened = F.normalize(flattened, p=2, dim=1)

 
    mbk = MiniBatchKMeans(
        n_clusters=num_cores,
        init='k-means++',
        random_state=42,
        batch_size=max(8192, num_cores),   
        max_iter=50,                       
        n_init=1,                          
        reassignment_ratio=0.01
    )
    mbk.fit(flattened.numpy())


    centers = torch.tensor(mbk.cluster_centers_, dtype=flattened.dtype)  
    distances = torch.cdist(flattened, centers, p=2)   
    core_indices = torch.argmin(distances, dim=0)      

    embeddings_tensor = embeddings_tensor[core_indices]
    indices_tensor    = indices_tensor[core_indices]

    return embeddings_tensor, indices_tensor



# Source code : https://github.com/decisionintelligence/CATCH/blob/master/ts_benchmark/baselines/catch/layers/RevIN.py
import torch
import torch.nn as nn

class RevIN1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, min_sigma: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return mu, sigma

    # def norm(self, x):
    #     self._mu, self._sigma = self._stats(x)
    #     x_hat = (x - self._mu) / self._sigma
    #     if self.affine:
    #         x_hat = x_hat * self.weight + self.bias
    #     return x_hat

    def norm(self, x):
        self._mu, self._sigma = self._stats(x)
        
        # with torch.no_grad():
        #     print(f"[RevIN] Before norm: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        #     print(f"[Before RevIN] mean per channel: {x.mean(dim=(0,2)).cpu().numpy()}")
            
        
        x_hat = (x - self._mu) / self._sigma
        
        if self.affine:
            x_hat = x_hat * self.weight + self.bias

        # with torch.no_grad():
        #     print(f"[RevIN] After norm : mean={x_hat.mean().item():.4f}, std={x_hat.std().item():.4f}")
        #     print(f"[After RevIN] mean per channel: {x_hat.mean(dim=(0,2)).cpu().numpy()}")
        #     print("-" * 60)
        
        return x_hat

    def denorm(self, x_hat):
        mu, sigma = self._mu, self._sigma
        if mu is None or sigma is None:
            raise RuntimeError("Call norm() before denorm().")
        if self.affine:
            w = self.weight if self.weight is not None else 1.0
            b = self.bias if self.bias is not None else 0.0
            x_hat = (x_hat - b) / (w + self.eps)
        return x_hat * sigma + mu
