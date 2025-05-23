import torch

def apply_pixeldp(activation, epsilon=1.0):
    laplace_noise = torch.distributions.Laplace(0, 1/epsilon)
    noise = laplace_noise.sample(activation.shape).to(activation.device)
    return activation + noise
