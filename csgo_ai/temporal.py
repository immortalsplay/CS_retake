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

class SinusoidalPosEmb(nn.Module): # 位置编码
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





def pair(t): # 用于将输入转换为元组
    return t if isinstance(t, tuple) else (t, t)



class PreNorm(nn.Module): # 用于在每个子层之前应用层归一化
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)





class FeedForward(nn.Module): # 前馈网络
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


class Attention(nn.Module): # 多头注意力机制
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
        # layer_norm_cfg = 1：在每个TransformerBlock中，仅在注意力（Attention）层之后应用层归一化（LayerNormalization）。
        # layer_norm_cfg = 2：在每个TransformerBlock中，仅在前馈网络（FeedForward）层之后应用层归一化（LayerNormalization）。
        # layer_norm_cfg = 3：在每个TransformerBlock中，不应用层归一化（LayerNormalization）。
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

class MultiClassInvDynamic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=-1)  # 用于多分类
        )

    def forward(self, x):
        return self.net(x)
    def predict(self, x):
        outputs = self.net(x)
        predicted = outputs.argmax(dim=-1)
        return predicted
class BinaryInvDynamic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class CSGO_model(nn.Module): # CSGO模型
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
        model_option = 1：在这种情况下，模型使用预训练的ResNet-50作为编码器。ResNet-50的全连接层被替换为一个新的线性层，其输出大小为num_feature。
        model_option = 2: 在这种情况下，模型使用预训练的ResNet-50作为编码器。全连接层被替换为一个新的线性层，其输出大小为num_feature。
        model_option = 3: 在这种情况下，模型使用预训练的ResNet-50作为编码器。ResNet-50的layer3和layer4被冻结，这意味着在训练过程中这些层的参数不会被更新。然后，全连接层被替换为一个新的线性层，其输出大小为1024。
        model_option = 4 或其他值：在这种情况下，模型使用未训练的ResNet-50作为编码器，其输出大小为num_feature。
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

        # 与后续网络连接
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

        # self.mouse_x_possibles= [ -140, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.00, 10.0, 20.0, 30.0, 60.0, 100.0, 140]
        # self.mouse_y_possibles= [-75, -60,  -50.00, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 60.0, 75]
        self.mouse_x_possibles = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                             9, 10]
        self.mouse_y_possibles = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                             9, 10]
        self.mouse_x_model = MultiClassInvDynamic(num_feature * self.frame_count, len(self.mouse_x_possibles), inverse_dynamic_dim)
        self.mouse_y_model = MultiClassInvDynamic(num_feature * self.frame_count, len(self.mouse_y_possibles), inverse_dynamic_dim)
        self.other_models = nn.ModuleList(
            [BinaryInvDynamic(num_feature * self.frame_count, 1, inverse_dynamic_dim) for _ in range(24)]
        )

    def forward(self, x):
        batch_size, horizon, _, _, _ = x.shape
        # print("batch:",batch_size)
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

        mouse_x_outputs = torch.zeros([batch_size,1,len(self.mouse_x_possibles)],device=x.device)
        mouse_y_outputs = torch.zeros([batch_size,1,len(self.mouse_y_possibles)],device=x.device)
        other_outputs = torch.zeros([batch_size, 24, 1], device=x.device)
        mouse_x_outputs = self.mouse_x_model(x_comb)
        mouse_y_outputs = self.mouse_y_model(x_comb)
        for b in range(batch_size):
            for i in range(24):
                temp = self.other_models[i](x_comb)
                # print("temp.shape:",temp.shape)
                # print(temp)
                other_outputs[b, i, 0] = temp[b]

        return mouse_x_outputs,mouse_y_outputs,other_outputs

    def predict(self, x):
        mouse_x_outputs, mouse_y_outputs, other_outputs = self.forward(x)

        # 解析鼠标移动预测,得到鼠标移动方向
        mouse_x_dir = mouse_x_outputs.argmax(dim=-1).item()
        mouse_y_dir = mouse_y_outputs.argmax(dim=-1).item()
        # mouse_x = self.mouse_x_possibles[mouse_x_dir]
        # mouse_y = self.mouse_y_possibles[mouse_y_dir]
        # 解析other_outputs,获得24个动作的预测
        action = []
        for output in other_outputs:
            if output > 0.5:
                action.append(1)
            else:
                action.append(0)
        return mouse_x_dir, mouse_y_dir, action



