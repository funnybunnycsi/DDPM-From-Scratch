import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_time_embedding(time_steps, t_emb_dim):
    assert t_emb_dim % 2 == 0, "Embedding dimension must be divisible by 2"
    factor = 10000 ** ((torch.arange(0, t_emb_dim//2, dtype=torch.float32, device=time_steps.device) / (t_emb_dim//2)))

    t_emb = time_steps[:, None].repeat(1, t_emb_dim//2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        self.residual_in_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)
        ])
        
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_1[i](x)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_2[i](out)
            out = out + self.residual_in_conv[i](resnet_input)

            batch_size, c, h, w = out.shape 
            in_attn = out.reshape(batch_size, c, h*w)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            out_attn, _ = self.attn[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1,2).reshape(batch_size, c, h, w)
            out = out + out_attn

        out = self.down_sample_conv(out)
        return out
    

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers+1)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers+1)
        ])

        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers+1)
        ])

        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        self.residual_in_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers+1)
        ])

    def forward(self, x, t_emb):
        resnet_input = x
        out = self.resnet_conv_1[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_2[0](out)
        out = out + self.residual_in_conv[0](resnet_input)

        for i in range(self.num_layers):
            batch_size, c, h, w = out.shape
            in_attn = out.reshape(batch_size, c, h*w)
            in_attn = self.attn_norm[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            out_attn, _ = self.attn[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1,2).reshape(batch_size, c, h, w)
            out = out + out_attn

            resnet_input = out
            out = self.resnet_conv_1[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_2[i+1](out)
            out = out + self.residual_in_conv[i+1](resnet_input)

        return out
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.up_sample = up_sample
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i==0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        self.residual_in_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)
        ])

        self.up_sample_conv = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=4, stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        # add multiple layers in future
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_1[i](x)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_2[i](out)
            out = out + self.residual_in_conv[i](resnet_input)

            batch_size, c, h, w = out.shape 
            in_attn = out.reshape(batch_size, c, h*w)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            out_attn, _ = self.attn[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1,2).reshape(batch_size, c, h, w)
            out = out + out_attn

        out = self.up_sample_conv(out)
        return out
    

class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_channels = cfg['img_channels']
        self.down_channels = cfg['down_channels']
        self.mid_channels = cfg['mid_channels']
        self.t_emb_dim = cfg['time_emb_dim']
        self.down_sample = cfg['down_sample']
        self.num_down_layers = cfg['num_down_layers']
        self.num_mid_layers = cfg['num_mid_layers']
        self.num_up_layers = cfg['num_up_layers']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.conv_in = nn.Conv2d(img_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, down_sample=self.down_sample[i], num_heads=self.num_heads, num_layers=self.num_down_layers))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.downs.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=self.num_heads, num_layers=self.num_mid_layers))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.downs.append(UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i!=0 else 16, self.t_emb_dim, up_sample=self.down_sample[i], num_heads=self.num_heads, num_layers=self.num_up_layers))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            print(out.shape)
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            print(out.shape)
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            print(out.shape, down_out.shape)
            out = up(out, down_out, t_emb)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out