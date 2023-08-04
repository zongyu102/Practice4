import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, depth=8, hidden_units=256, position_ch=64,
                 direction_ch=24, output_ch=4, skip=4, use_viewdirs=True):

        super().__init__()
        self.position_ch = position_ch
        self.direction_ch = direction_ch
        self.skip = skip
        self.output_ch = output_ch
        self.use_viewdirs = use_viewdirs
        #处理前8个全连接层
        self.linears = nn.ModuleList([nn.Linear(position_ch, hidden_units)])
        for i in range(depth - 1):
            if i == skip:#位置信息和输出拼接
                self.linears.append(nn.Linear(hidden_units + position_ch, hidden_units))
            else:
                self.linears.append(nn.Linear(hidden_units, hidden_units))

        self.sigma_layer = nn.Linear(hidden_units, 1)#体密度
        self.feature_linear = nn.Linear(hidden_units, hidden_units)#特征向量

        if self.use_viewdirs:#加入方向信息
            self.view_linears = nn.Linear(hidden_units + direction_ch, hidden_units // 2)
        else:
            self.view_linears = nn.Linear(hidden_units, hidden_units // 2)
        self.color_layer = nn.Linear(hidden_units // 2, 3)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        input_pos, input_dir = torch.split(inputs, [self.position_ch, self.direction_ch], dim=-1)
        x = input_pos
        for idx, layer in enumerate(self.linears):
            x = self.relu(layer(x))
            if idx == self.skip:
                x = torch.cat([input_pos, x], dim=-1)

        sigma = self.sigma_layer(x)
        x = self.feature_linear(x)
        if self.use_viewdirs:
            x = torch.cat([x, input_dir], dim=-1)
        x = self.view_linears(x)
        x = self.relu(x)
        color = self.color_layer(x)
        outputs = torch.cat([color, sigma], dim=-1)
        return outputs