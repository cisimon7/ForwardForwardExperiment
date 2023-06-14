import math
import torch as th


def mlp_block(input_dim, output_dims, p):
    return th.nn.Sequential(
        th.nn.Linear(input_dim, output_dims, bias=True),
        th.nn.ReLU(True),
        th.nn.Dropout(p=p)
    )


# Has no parameters, just structured this way for convenience
class PatchMakerModule(th.nn.Module):
    def __init__(self, patch_length, patch_width):
        super(PatchMakerModule, self).__init__()
        self.patcher = th.nn.Unfold(
            kernel_size=(patch_length, patch_width),
            dilation=1,
            stride=(patch_length, patch_width)
        )

    def forward(self, images):  # N x 3 x image_size x image_size
        patches = self.patcher(images)  # N x (patch_size・patch_size・3) x num_patches
        return patches.transpose(-1, -2)  # N x num_patches x (patch_size・patch_size・3)
    
    
class PatchEncoderFF(th.nn.Module):
    def __init__(
            self,
            num_patches, patch_size, projection_dim,
            loss_func=th.nn.BCEWithLogitsLoss(), ratio_peer=0, device=None
    ):
        super(PatchEncoderFF, self).__init__()
        self.positions = th.arange(start=0, end=num_patches, step=1, device=device)
        self.projection = th.nn.Linear(
            (patch_size * patch_size * 3) + 10, projection_dim, bias=False
        )
        self.feedback = th.nn.Linear(
            projection_dim, 10, bias=False
        )  
        self.embedding = th.nn.Embedding(
            num_embeddings=num_patches,
            embedding_dim=projection_dim
        )
        self.loss, self.peer_loss = None, None
        self.peer_mean = th.zeros(projection_dim, device=device) + 0.5
        self.loss_func, self.ratio_peer = loss_func, ratio_peer
        self.setup()
    
    def setup(self):
        th.nn.init.normal_(
            self.projection.weight,
            mean=0, std=1/math.sqrt(self.projection.weight.shape[0])
        )
        th.nn.init.normal_(
            self.embedding.weight,
            mean=0, std=1/math.sqrt(self.embedding.weight.shape[0])
        )
    
    def set_loss_peer(self, z, labels):
        # x = self.feedback(z)
        logits = th.sum(th.flatten(z**2, start_dim=1), dim=-1, keepdim=True) - z.shape[1]
        self.loss = self.loss_func(logits, labels)  # Equation 1
        
        gamma = 0.9
        positive_idx = th.nonzero(labels == 1)
        peer_mean = th.mean(z[positive_idx], dim=0)
        self.peer_mean = (gamma * self.peer_mean.detach()) + (peer_mean * (1 - gamma))
        self.peer_loss = th.mean((th.mean(self.peer_mean) - self.peer_mean) ** 2)
        self.loss += (self.peer_loss * self.ratio_peer)

        return self.loss, self.peer_loss

    def forward(self, patches, labels):
        x = self.projection(patches) + self.embedding(self.positions)
        
        if self.training:
            self.set_loss_peer(x, labels)
        
        x = x.detach()
        x = th.nn.functional.layer_norm(x, x.size())
        return x
    

class TransformerBlockFF(th.nn.Module):
    def __init__(
            self,
            num_patches, projection_dim, embed_dim, num_heads, kdim=None,
            loss_func=th.nn.BCEWithLogitsLoss()
    ):
        super(TransformerBlockFF, self).__init__()
        self.layer1 = th.nn.LayerNorm(normalized_shape=(num_patches, projection_dim), eps=1e-6)
        self.layer2 = th.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            kdim=kdim, dropout=0.1, batch_first=True
        )
        self.layer3 = th.nn.Sequential(
            th.nn.LayerNorm(normalized_shape=(num_patches, projection_dim), eps=1e-6),
            th.nn.Sequential(*(
                mlp_block(in_dim, out_dim, 0.1)
                for (in_dim, out_dim) in th.tensor(
                    [projection_dim, 2*projection_dim, projection_dim]
                ).unfold(0, 2, 1)
            ))
        )
        self.feedback = th.nn.Linear(
            projection_dim, 10, bias=False
        )
        self.loss_func = loss_func
        self.loss = None
    
    def set_loss(self, z, labels):
        # x = self.feedback(z)
        logits = th.sum(th.flatten(z**2, start_dim=1), dim=-1, keepdim=True) - z.shape[1]
        self.loss = self.loss_func(logits, labels)  # Equation 1

        return self.loss

    def forward(self, x, labels):
        x1 = self.layer1(x)
        attn_output, attn_output_weights = self.layer2(x1, x1, x1)
        x2 = attn_output + x
        x3 = self.layer3(x2)
        
        if self.training:
            self.set_loss(x, labels)
            
        x = x3 + x2
        x.detach()
        x = th.nn.functional.layer_norm(x, x.size())
        return x
    