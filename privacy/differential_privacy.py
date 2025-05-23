import torch

def apply_dp_to_gradients(gradients, l2_norm_clip=1.0, noise_multiplier=1.1):
    noisy_gradients = []
    for grad in gradients:
        norm = torch.norm(grad)
        if norm > l2_norm_clip:
            grad = grad * (l2_norm_clip / norm)
        noise = torch.randn_like(grad) * noise_multiplier
        grad = grad + noise
        noisy_gradients.append(grad)
    return noisy_gradients
