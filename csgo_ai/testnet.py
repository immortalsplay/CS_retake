import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
import math
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################################################################
#                                Transformer Model                              #
#################################################################################

class SinusoidalPosEmb(nn.Module): # location encoder
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb





def pair(t): # tuple
    return t if isinstance(t, tuple) else (t, t)



class PreNorm(nn.Module): # normalization
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)





class FeedForward(nn.Module): # FF
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # dim=1024, hidden_dim=2048
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): # Muti attention
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): # x: [batch_size, 20, 1024]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_block(nn.Module): # Transformer Block
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, layer_norm_cfg, dropout=0.):
        # layer_norm_cfg = 1:TransformerBlock Attention LayerNormalization
        # layer_norm_cfg = 2:TransformerBlock FeedForward LayerNormalization 
        # layer_norm_cfg = 3:TransformerBlock LayerNormalization
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if layer_norm_cfg == 1:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    nn.Identity(),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]))
            elif layer_norm_cfg == 2:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.Identity(),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
            else:  # 3:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.Identity(),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]))

    def forward(self, x):
        """
        x: batch * horizon * dim
        """
        for attn, norm, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x
        return x


#################################################################################
#                                   CSGO Model                                  #
#################################################################################

class InvDynamic(nn.Module): # Inverse Kinematic,mapping to 26 moves
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CSGO_model(nn.Module): # CSGOmodel
    def __init__(self, horizon, num_feature, depth, num_heads, head_dim, inverse_dynamic_dim, layer_norm_cfg,
                 model_option,frame_count):
        """

        :param horizon:
        :param num_feature:
        :param depth:
        :param num_heads:
        :param head_dim:
        :param inverse_dynamic_dim:
        :param layer_norm_cfg:
        :param model_option:
        :param frame_count:
        model_option = 1: ResNet-50 as the encoder. The fully connected layer of ResNet-50 is replaced with a new linear layer, whose output size is num_feature.
        model_option = 2: ResNet-50 as the encoder. The fully connected layer is replaced with a new linear layer, whose output size is num_feature.
        model_option = 3: In this case, the model uses a pretrained ResNet-50 as the encoder. The layer3 and layer4 of ResNet-50 are frozen, which means that the parameters of these layers will not be updated during the training process. Then, the fully connected layer is replaced with a new linear layer, whose output size is 1024.
        model_option = 4 ResNet-50 as encode and save num_feature。
        """
        self.num_feature=num_feature
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = resnet
        self.model_option = model_option
        self.frame_count=frame_count
        if model_option == 1:
            resnet = models.resnet50(pretrained=True)
        elif model_option == 2:
            resnet = models.resnet50(pretrained=True)
            self.encoder.fc = nn.Linear(100352, num_feature)
        elif model_option == 3:
            resnet = models.resnet50(pretrained=True)
            for param in resnet.layer3.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = False
            resnet.fc = nn.Linear(resnet.fc.in_features, 1024)
        else:
            resnet = models.resnet50(pretrained=False)

        self.encoder = resnet

        # connection
        if model_option == 1:
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, num_feature)
        elif model_option == 3 :
            self.encoder.fc = nn.Linear(2048, num_feature)
        else:
            pass
        self.transformer = Transformer_block(dim=num_feature,
                                             depth=depth,
                                             heads=num_heads,
                                             dim_head=head_dim,
                                             mlp_dim=2048,
                                             layer_norm_cfg=layer_norm_cfg,
                                             dropout=0.
                                             )

        self.fc_outputs = nn.ModuleList(
            [InvDynamic(num_feature * self.frame_count, 1, inverse_dynamic_dim) for _ in range(26)]
        )

    def forward(self, x):
        batch_size, horizon, _, _, _ = x.shape

        batch_size = x.shape[0]

        x = rearrange(x, 'b h c x y -> (b h) c x y')

        if self.model_option == 1 or self.model_option == 4:
            x = self.encoder(x)
        elif self.model_option == 2 :
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

            if self.model_option == 2:
                self.encoder.fc = nn.Linear(100352, self.num_feature).to(device)
                x = self.encoder.layer3(x)
                x = self.encoder.layer4(x)
            x = torch.flatten(x, 1)
            x = self.encoder.fc(x).to(device)
        else:
            x = self.encoder(x)
            # print("test2_x.shape:", x.shape)
        x = rearrange(x, '(b h) d -> b h d', b=batch_size)
        # print("test3_x.shape:", x.shape)

        x = self.transformer(x)
        concatenated_frames = []
        for i in range(self.frame_count):
            concatenated_frames.append(x[:, i, :])
        x_comb = torch.cat(concatenated_frames, dim=1)


        outputs = torch.zeros([batch_size, 26, 1], device=x.device)

        for b in range(batch_size):
            i = 0
            for fc in self.fc_outputs:
                # print(fc(x_comb).shape)
                # print("____")
                # print(outputs[b, i].shape)
                outputs[b, i] = fc(x_comb[b])
                i += 1

        return outputs
