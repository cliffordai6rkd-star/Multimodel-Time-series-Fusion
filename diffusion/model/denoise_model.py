import torch
import torch.nn as nn



class Denoiser(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        config = config or {}
        self.diffusion_config = config.get("diffusion",{})

        self.num_classes = int(self.diffusion_config.get("num_classes", 4))
        self.num_steps = int(self.diffusion_config.get("num_steps", 100))
        self.hidden_channels = int(self.diffusion_config.get("hidden_channels",64))
        self.time_embed = nn.Embedding(self.num_steps, self.hidden_channels)
        self.label_embed = nn.Embedding(self.num_classes, self.hidden_channels) # 编码成向量

        self.input_layer = nn.Sequential(
            nn.Conv1d(2, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),)
        self.condition_fuse_layer = nn.Sequential(
            nn.Conv1d(self.hidden_channels * 2, self.hidden_channels, kernel_size=1),
            nn.ReLU(),)
        self.middle_layer = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),)
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.hidden_channels, 2, kernel_size=3, padding=1))
        


    def forward(self, x, step_t, label):
        feature = self.input_layer(x) # x(B,2,1024) --> feature(B,64,1024)


        t_embed = self.time_embed(step_t) # step_t[B] -->t-embed [B,64]
        label_embed = self.label_embed(label)
        cond = t_embed + label_embed # # 将 timestep 条件和 label 条件相加，得到一个全局条件向量
        condition_cond = cond.unsqueeze(-1) # condition_cond [B,64,1]
        # print(condition_cond.shape)
        # print(feature.shape) 
        cond = condition_cond.expand(-1, -1, feature.shape[-1])# condition_cond [B,64,1024]维度对齐
        # print(cond.shape)  
         # 将条件特征与数据特征中分别提取的特征融合  
        #将全局条件扩展到每个时间位置，再与数据特征在通道维拼接
        # 每个时间点都会获得一个time_step、label条件
        fused_input = torch.cat([feature, cond], dim=1) # 拼接[B, 128, 1024]
        # print(fused_input.shape)


        feature = self.condition_fuse_layer(fused_input)    # [B, 128, 1024] -> [B, 64, 1024]
        #把数据特征 + 条件特征融合回 64 通道
        feature = self.middle_layer(feature)

        pred_noise = self.output_layer(feature) # feature(B,64,1024) --> x(B,2,1024)

        return pred_noise


if __name__ == "__main__":
    x = torch.randn(4, 2, 1024)
    model = Denoiser()
    step_t = torch.randint(0, 100, (4,))
    label = torch.randint(0, 4, (4,))

    y = model(x, step_t, label)

    print(y.shape)
