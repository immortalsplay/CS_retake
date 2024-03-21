import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


#################################################################################
#                                    ViT Model                                  #
#################################################################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
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


class Attention(nn.Module):
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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


def create_fc_layers(input_dim, output_dim, hidden_layers, activation=nn.LeakyReLU, batch_norm=False):
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation(0.1))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

#################################################################################
#                                   CSGO Model                                  #
#################################################################################

class CSGO_model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.vit = ViT(image_size = input_shape,
        patch_size = 10,
        num_classes = 200,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1)


        self.TimeDistributed_block = TimeDistributed(self.vit, batch_first=False)
        hidden_layers = [200, 200, 200]
        self.fc_aim_x = create_fc_layers(200, 45, hidden_layers, batch_norm=True)
        self.fc_aim_y = create_fc_layers(200, 57, hidden_layers)

        self.fc_w = create_fc_layers(200, 2, hidden_layers)
        self.fc_a = create_fc_layers(200, 2, hidden_layers)
        self.fc_s = create_fc_layers(200, 2, hidden_layers)
        self.fc_d = create_fc_layers(200, 2, hidden_layers)

        self.fc_fire = create_fc_layers(200, 2, hidden_layers, batch_norm=True)
        self.fc_scope = create_fc_layers(200, 2, hidden_layers)
        self.fc_jump = create_fc_layers(200, 2, hidden_layers)
        self.fc_crouch = create_fc_layers(200, 2, hidden_layers)
        self.fc_walking = create_fc_layers(200, 2, hidden_layers)
        self.fc_reload = create_fc_layers(200, 2, hidden_layers)
        self.fc_e = create_fc_layers(200, 2, hidden_layers)
        self.fc_switch = create_fc_layers(200, 6, hidden_layers)

    def forward(self, x):
        x = self.vit(x)
        x = self.TimeDistributed_block(x)

        w = self.fc_w(x)
        a = self.fc_a(x)
        s = self.fc_s(x)
        d = self.fc_d(x)
        fire = self.fc_fire(x)
        scope = self.fc_scope(x)
        jump = self.fc_jump(x)
        crouch = self.fc_crouch(x)
        walking = self.fc_walking(x)
        reload = self.fc_reload(x)
        e = self.fc_e(x)
        switch = self.fc_switch(x)
        aim_x = self.fc_aim_x(x)
        aim_y = self.fc_aim_y(x)

        return w, a, s, d, fire, scope, jump, \
            crouch, walking, reload, e, switch, \
            aim_x, aim_y

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=1e-4):
        # 将数据转换为张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        # 训练模型
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                # 将数据移动到GPU上
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 前向传播
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 验证模型
            self.eval()
            val_loss = 0.0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)

                val_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f} - Val_loss: {val_loss / len(val_loader):.4f}")
