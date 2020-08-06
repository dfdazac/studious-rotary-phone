import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from torch.distributions import MultivariateNormal


class MultivariateNormalMixture:
    def __init__(self, means, covariances):
        self.p1 = MultivariateNormal(means[0], covariances[0])
        self.p2 = MultivariateNormal(means[1], covariances[1])
        
    def log_prob(self, values):
        p1 = 0.5 * self.p1.log_prob(values).exp()
        p2 = 0.5 * self.p2.log_prob(values).exp()
        return torch.log(p1 + p2)
    
    def sample(self, num_samples):
        samples = torch.zeros([num_samples, 2])
        idx = torch.randint(0, 2, [num_samples], dtype=torch.bool)
        samples[idx] = self.p1.sample([num_samples])[idx]
        samples[~idx] = self.p2.sample([num_samples])[~idx]
        return samples

    
def make_mesh():
    domain = torch.linspace(-3, 3, 50)
    u1, u2 = torch.meshgrid((domain, domain))
    u = torch.stack((u1, u2), dim=-1)
    return u1, u2, u


def plot_surface(u1, u2, p_u):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(u1.numpy(), u2.numpy(), p_u.numpy(),
                           cmap='viridis', edgecolors='k', lw=0.4)
    ax.set_zlim((0, p_u.max().item() * 2.5))
    ax.view_init(15, 135)
    
    
def plot_density(distribution):
    # Make data.
    u1, u2, u = make_mesh()
    p_u = distribution.log_prob(u).exp()
    
    plot_surface(u1, u2, p_u)    

    
base = MultivariateNormal(torch.zeros(2), torch.eye(2))
target = MultivariateNormalMixture([1.5 * torch.ones(2), -1.5 * torch.ones(2)],
                                   [0.2 * torch.eye(2)] * 2)