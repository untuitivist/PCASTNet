import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from function import calc_mean_std, reconstruct_gray_image, batch_reconstruct_gray

#----网络参数----

num_classes = 4

#---------------

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),  # 0
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 1
    nn.Conv2d(3, 64, (3, 3)),  # 2
    nn.ReLU(),  # 3 relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(64, 64, (3, 3)),  # 5
    nn.ReLU(),  # 6 relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 7
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 8
    nn.Conv2d(64, 128, (3, 3)),  # 9
    nn.ReLU(),  # 10 relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 11
    nn.Conv2d(128, 128, (3, 3)),  # 12
    nn.ReLU(),  # 13 relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 14
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 15
    nn.Conv2d(128, 256, (3, 3)),  # 16
    nn.ReLU(),  # 17 relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 18
    nn.Conv2d(256, 256, (3, 3)),  # 19
    nn.ReLU(),  # 20 relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 21
    nn.Conv2d(256, 256, (3, 3)),  # 22
    nn.ReLU(),  # 23 relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(256, 256, (3, 3)),  # 25
    nn.ReLU(),  # 26 relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 27
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 28
    nn.Conv2d(256, 512, (3, 3)),  # 29
    nn.ReLU(),  # 30 relu4-1, this is the last layer used to style transfer
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 31
    nn.Conv2d(512, 512, (3, 3)),  # 32
    nn.ReLU(),  # 33 relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 34
    nn.Conv2d(512, 512, (3, 3)),  # 35
    nn.ReLU(),  # 36 relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 37
    nn.Conv2d(512, 512, (3, 3)),  # 38
    nn.ReLU(),  # 39 relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 40
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 41
    nn.Conv2d(512, 512, (3, 3)),  # 42
    nn.ReLU(),  # 43 relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 44
    nn.Conv2d(512, 512, (3, 3)),  # 45
    nn.ReLU(),  # 46 relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 47
    nn.Conv2d(512, 512, (3, 3)),  # 48
    nn.ReLU(),  # 49 relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 50
    nn.Conv2d(512, 512, (3, 3)),  # 51
    nn.ReLU()  # 52 relu5-4
)

classifier = lambda num_classes: nn.Sequential(
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
)

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho_c'):
            module.rho_c.data = module.rho_c.data.clamp(self.clip_min, self.clip_max)
        if hasattr(module, 'rho_s'):
            module.rho_s.data = module.rho_s.data.clamp(self.clip_min, self.clip_max)

class AdaLIN(nn.Module):
    def __init__(self, channels, rho_c=None, rho_s=None, epsilon=1e-5):
        super(AdaLIN, self).__init__()
        print("[+] Build AdaLIN")
        self.epsilon = epsilon if epsilon is not None else 1e-5  # 确保 epsilon 有默认值

        # 处理 rho_c
        if rho_c is None:
            self.rho_c = nn.Parameter(torch.ones(1, channels, 1, 1), requires_grad=True)  # 需要训练
            self.rho_c.data.fill_(0.9)
            print(" - rho_c is None, using default value {} and requires grad {}".format(self.rho_c.data.mean().item(), self.rho_c.requires_grad))
        else:
            # 确保 rho_c 被扩展成正确的形状
            self.register_buffer("rho_c", torch.full((1, channels, 1, 1), rho_c, dtype=torch.float32))  # 使用常数填充
            print(" - rho_c is not None, using default value {} and requires grad {}".format(self.rho_c.data.mean().item(), self.rho_c.requires_grad))

        # 处理 rho_s
        if rho_s is None:
            self.rho_s = nn.Parameter(torch.ones(1, channels, 1, 1), requires_grad=True)  # 需要训练
            self.rho_s.data.fill_(0.9)
            print(" - rho_s is None, using default value {} and requires grad {}".format(self.rho_s.data.mean().item(), self.rho_s.requires_grad))
        else:
            # 确保 rho_s 被扩展成正确的形状
            self.register_buffer("rho_s", torch.full((1, channels, 1, 1), rho_s, dtype=torch.float32))  # 使用常数填充
            print(" - rho_s is not None, using default value {} and requires grad {}".format(self.rho_s.data.mean().item(), self.rho_s.requires_grad))

    def instance_norm(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.epsilon
        return (x - mean) / std
    
    def layer_norm(self, x):
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True) + self.epsilon
        return (x - mean) / std
    
    def forward(self, content, style):
        c_in, c_ln = self.instance_norm(content), self.layer_norm(content)
        s_mean_in, s_std_in = style.mean(dim=[2, 3], keepdim=True), style.std(dim=[2, 3], keepdim=True) + self.epsilon
        s_mean_ln, s_std_ln = style.mean(dim=[1, 2, 3], keepdim=True), style.std(dim=[1, 2, 3], keepdim=True) + self.epsilon
        
        norm_c = self.rho_c * c_in + (1 - self.rho_c) * c_ln
        norm_s = self.rho_s * s_std_in + (1 - self.rho_s) * s_std_ln
        shift_s = self.rho_s * s_mean_in + (1 - self.rho_s) * s_mean_ln
        return norm_s * norm_c + shift_s


class ADAILNet(nn.Module):
    def __init__(self, encoder, adailn, decoder):
        super(ADAILNet, self).__init__()
        self.encoder = encoder
        self.adailn = adailn
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        self.make_encoder()

    def make_encoder(self):
        enc_layers = list(self.encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:])  # relu4_1 -> relu5_4

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def calc_perceptual_loss(self, g_t_feats, style_feats):
        """ 计算感知损失 """
        perceptual_loss = 0
        # 选择高层特征进行感知损失计算，例如 relu3_1 (index 2) 和 relu4_1 (index 3)
        selected_layers = [2, 3]
        for i in selected_layers:
            perceptual_loss += F.l1_loss(g_t_feats[i], style_feats[i])
        return perceptual_loss
    
    def calc_total_variation_loss(self, g_t):
        """
        计算图像的总变分损失 (Total Variation Loss)
        """
        tv_loss = (torch.mean(torch.abs(g_t[:, :, :-1, :] - g_t[:, :, 1:, :])) + \
                  torch.mean(torch.abs(g_t[:, :, :, :-1] - g_t[:, :, :, 1:])))/2
        return tv_loss
    

    def calc_energe_loss(self, g_t, style):
        """批量计算能量损失"""
        # 输入形状：g_t/style -> (B,C,H,W)
        gray_g = batch_reconstruct_gray(g_t)  # (B,H,W)
        gray_s = batch_reconstruct_gray(style) # (B,H,W)
        
        # 确保数据类型一致
        gray_g = gray_g.to(torch.float32)
        gray_s = gray_s.to(torch.float32)

        # 总能量为灰度图像系数之和
        # energe_g = torch.sum(gray_g, dim=(1, 2))  # (B,)
        # energe_s = torch.sum(gray_s, dim=(1, 2))  # (B,)

        # 频带能量为灰度图像每个条带系数之和
        energe_g = torch.sum(gray_g, dim=1)  # (B,W)
        energe_s = torch.sum(gray_s, dim=1)  # (B,W)

        
        return F.l1_loss(energe_g, energe_s)
    
    def forward(self, content, style):
        """ 执行风格迁移任务 """
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)  # 这里必须是 encode 而非 encode_with_intermediate

        t = self.adailn(content_feat, style_feats[-1])
        g_t = self.decoder(t)

        g_t_feats = self.encode_with_intermediate(g_t)

        # 计算内容损失
        loss_c = self.calc_content_loss(g_t_feats[-1], t)

        # 计算风格损失
        loss_s = sum(self.calc_style_loss(g_t_feats[i], style_feats[i]) for i in range(4))

        # 计算感知损失
        loss_p = self.calc_perceptual_loss(g_t_feats, style_feats)

        # 计算总变差损失
        loss_v = self.calc_total_variation_loss(g_t)

        # 计算能量损失
        loss_e = self.calc_energe_loss(g_t, style)

        return g_t, loss_c, loss_s, loss_p, loss_v, loss_e
    
    def save(self, save_pth):
        torch.save(self.encoder.state_dict(), save_pth.replace(".pth", "_encoder.pth"))
        torch.save(self.adailn.state_dict(), save_pth.replace(".pth", "_adailn.pth"))
        torch.save(self.decoder.state_dict(), save_pth.replace(".pth", "_decoder.pth"))
        print(' - Model saved at', save_pth)

def build_ADAILNet_module(num_classes, 
        adailn_channels=512, 
        rho_c=None, 
        rho_s=None, 
        encoder_pth=None, 
        adailn_pth=None, 
        decoder_pth=None, 
        Rho_clipper=RhoClipper(0, 1)
        ): 

    net = ADAILNet(vgg, 
                AdaLIN(adailn_channels, rho_c, rho_s),
                decoder
                )

    if encoder_pth:
        net.encoder.load_state_dict(torch.load(encoder_pth))
        net.make_encoder()
    if adailn_pth:
        net.adailn.load_state_dict(torch.load(adailn_pth))
    if decoder_pth:
        net.decoder.load_state_dict(torch.load(decoder_pth))
    
    net.adailn.apply(Rho_clipper)

    return net


def build_ADAILNet_overall(num_classes, 
        adailn_channels=512, 
        rho_c=None, 
        rho_s=None, 
        net_pth=None,
        Rho_clipper=RhoClipper(0, 1)
        ): 
    
    net = ADAILNet(vgg, 
                AdaLIN(adailn_channels, rho_c, rho_s), 
                decoder
                )

    if net_pth:
        net.load_state_dict(torch.load(net_pth))
        net.make_encoder()

    net.adailn.apply(Rho_clipper)

    
    return net


