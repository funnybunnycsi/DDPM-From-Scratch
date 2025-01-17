import torch

class LinearNoiseScheduler:
    def __init__(self, T, beta_1, beta_T):
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        
        self.betas = torch.linspace(self.beta_1, self.beta_T, T)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def add_noise(self, original:torch.Tensor, noise, t):
        """
        Forward process for diffusion
        Args:
            original: Image tensor of shape (B, C, H, W)
            noise: Random noise tensor of same shape as original
            t: Timestep indices of shape (B,) (B->batch_size == T)
        Returns:
            Noised image tensor of same shape as original
        """

        sqrt_alpha_bar = self.sqrt_alpha_bars.to(original.device)[t, None, None, None] # (B, 1, 1, 1)
        sqrt_1_minus_alpha_bar = self.sqrt_1_minus_alpha_bars.to(original.device)[t, None, None, None] # (B)

        return sqrt_alpha_bar*original + sqrt_1_minus_alpha_bar*noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - self.sqrt_1_minus_alpha_bars[t]*noise_pred)/self.sqrt_alpha_bars[t]
        x0 = torch.clamp(x0,-1., 1.) # change to normalization in future as an experiment

        mean = (xt - ((self.betas[t]*noise_pred)/self.sqrt_1_minus_alpha_bars[t])) / torch.sqrt(self.alphas)

        if t==0:
            return mean, x0
        else:
            var = self.betas[t] * (1-self.alpha_bars[t-1]) / (1-self.alpha_bars[t])
            sigma = var ** 0.5
            
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0

