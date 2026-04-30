import torch
import numpy as np
import torch.nn.functional as F

def compute_reconstruction_errors(model, X):
    errors = []

    model.eval()
    with torch.no_grad():
        for x in X:
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0)

            output = model(x)
            error = F.mse_loss(output, x).item()

            errors.append(error)

    return errors


def compute_latent_center(model, X):
    embeddings = []

    model.eval()
    with torch.no_grad():
        for x in X:
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0)
            embeddings.append(model.encode(x).view(-1))

    return torch.stack(embeddings).mean(dim=0)


def compute_latent_distances(model, X, center):
    distances = []

    model.eval()
    with torch.no_grad():
        for x in X:
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0)
            z = model.encode(x).view(-1)
            distances.append(torch.norm(z - center, p=2).item())

    return distances


def summarize_file_components(model, file_features, latent_center, top_k=3):
    components = []

    for features in file_features:
        recon_errors = compute_reconstruction_errors(model, features)
        latent_distances = compute_latent_distances(model, features, latent_center)

        k = min(top_k, len(recon_errors))
        components.append(
            {
                "recon": np.mean(sorted(recon_errors, reverse=True)[:k]),
                "latent": np.mean(sorted(latent_distances, reverse=True)[:k]),
            }
        )

    return components


def fit_component_scaler(components):
    recon = np.array([item["recon"] for item in components])
    latent = np.array([item["latent"] for item in components])
    return {
        "recon_mean": recon.mean(),
        "recon_std": recon.std() + 1e-8,
        "latent_mean": latent.mean(),
        "latent_std": latent.std() + 1e-8,
    }


def combine_file_scores(components, scaler, recon_weight=1.0, latent_weight=1.0):
    scores = []

    for item in components:
        recon_score = (item["recon"] - scaler["recon_mean"]) / scaler["recon_std"]
        latent_score = (item["latent"] - scaler["latent_mean"]) / scaler["latent_std"]
        score = (recon_weight * recon_score) + (latent_weight * latent_score)
        scores.append(score)

    return scores


def compute_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)
