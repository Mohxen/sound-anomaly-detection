import torch
import numpy as np

def compute_error(model, X, center=None):
    errors = []

    model.eval()
    with torch.no_grad():
        for x in X:
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0)

            z = model.encode(x)
            z = z.view(-1)

            if center is not None:
                error = torch.norm(z - center, p=2).item()
            else:
                # fallback (old)
                error = torch.norm(z, p=2).item()

            errors.append(error)

    return errors


def compute_threshold(errors):
    return np.percentile(errors, 75)