import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        self.slide_winsize = in_channels * kernel_size * kernel_size

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = ((output - output_bias) * self.slide_winsize) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class Inpaint_Depth_Net(nn.Module):
    def __init__(self, layer_size=7, upsampling_mode='nearest'):
        super().__init__()
        in_channels = 4
        out_channels = 1
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample='down-7', conv_bias=True)
        self.enc_2 = PCBActiv(64, 128, sample='down-5', conv_bias=True)
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + in_channels, out_channels,
                              bn=False, activ=None, conv_bias=True)
    def add_border(self, input, mask_flag, PCONV=True):
        with torch.no_grad():
            h = input.shape[-2]
            w = input.shape[-1]
            require_len_unit = 2 ** self.layer_size
            residual_h = int(np.ceil(h / float(require_len_unit)) * require_len_unit - h) # + 2*require_len_unit
            residual_w = int(np.ceil(w / float(require_len_unit)) * require_len_unit - w) # + 2*require_len_unit
            enlarge_input = torch.zeros((input.shape[0], input.shape[1], h + residual_h, w + residual_w)).to(input.device)
            if mask_flag:
                if PCONV is False:
                    enlarge_input += 1.0
                enlarge_input = enlarge_input.clamp(0.0, 1.0)
            else:
                enlarge_input[:, 2, ...] = 0.0
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input

        return enlarge_input, [anchor_h, anchor_h+h, anchor_w, anchor_w+w]

    def forward_3P(self, mask, context, depth, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((depth, edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h)
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w)
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            # enlarge_input[:, 3] = 1. - enlarge_input[:, 3]
            depth_output = self.forward(enlarge_input)
            depth_output = depth_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]
            # import pdb; pdb.set_trace()

        return depth_output

    def forward(self, input_feat, refine_border=False, sample=False, PCONV=True):
        input = input_feat
        input_mask = (input_feat[:, -2:-1] + input_feat[:, -1:]).clamp(0, 1).repeat(1, input.shape[1], 1, 1)

        vis_input = input.cpu().data.numpy()
        vis_input_mask = input_mask.cpu().data.numpy()
        H, W = input.shape[-2:]
        if refine_border is True:
            input, anchor = self.add_border(input, mask_flag=False)
            input_mask, anchor = self.add_border(input_mask, mask_flag=True, PCONV=PCONV)
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
        output = h
        if refine_border is True:
            h_mask = h_mask[..., anchor[0]:anchor[1], anchor[2]:anchor[3]]
            output = output[..., anchor[0]:anchor[1], anchor[2]:anchor[3]]

        return output

class Inpaint_Edge_Net(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(Inpaint_Edge_Net, self).__init__()
        in_channels = 7
        out_channels = 1
        self.encoder = []
        # 0
        self.encoder_0 = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0), True),
                            nn.InstanceNorm2d(64, track_running_stats=False),
                            nn.ReLU(True))
        # 1
        self.encoder_1 = nn.Sequential(
                            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), True),
                            nn.InstanceNorm2d(128, track_running_stats=False),
                            nn.ReLU(True))
        # 2
        self.encoder_2 = nn.Sequential(
                            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), True),
                            nn.InstanceNorm2d(256, track_running_stats=False),
                            nn.ReLU(True))
        # 3
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)
        # + 3
        self.decoder_0 = nn.Sequential(
                            spectral_norm(nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, stride=2, padding=1), True),
                            nn.InstanceNorm2d(128, track_running_stats=False),
                            nn.ReLU(True))
        # + 2
        self.decoder_1 = nn.Sequential(
                            spectral_norm(nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1), True),
                            nn.InstanceNorm2d(64, track_running_stats=False),
                            nn.ReLU(True))
        # + 1
        self.decoder_2 = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(in_channels=64+64, out_channels=out_channels, kernel_size=7, padding=0),
                            )

        if init_weights:
            self.init_weights()

    def add_border(self, input, channel_pad_1=None):
        h = input.shape[-2]
        w = input.shape[-1]
        require_len_unit = 16
        residual_h = int(np.ceil(h / float(require_len_unit)) * require_len_unit - h) # + 2*require_len_unit
        residual_w = int(np.ceil(w / float(require_len_unit)) * require_len_unit - w) # + 2*require_len_unit
        enlarge_input = torch.zeros((input.shape[0], input.shape[1], h + residual_h, w + residual_w)).to(input.device)
        if channel_pad_1 is not None:
            for channel in channel_pad_1:
                enlarge_input[:, channel] = 1
        anchor_h = residual_h//2
        anchor_w = residual_w//2
        enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input

        return enlarge_input, [anchor_h, anchor_h+h, anchor_w, anchor_w+w]

    def forward_3P(self, mask, context, rgb, disp, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((rgb, disp/disp.max(), edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h)
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w)
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            edge_output = self.forward(enlarge_input)
            edge_output = edge_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]

        return edge_output

    def forward(self, x, refine_border=False):
        if refine_border:
            x, anchor = self.add_border(x, [5])
        x1 = self.encoder_0(x)
        x2 = self.encoder_1(x1)
        x3 = self.encoder_2(x2)
        x4 = self.middle(x3)
        x5 = self.decoder_0(torch.cat((x4, x3), dim=1))
        x6 = self.decoder_1(torch.cat((x5, x2), dim=1))
        x7 = self.decoder_2(torch.cat((x6, x1), dim=1))
        x = torch.sigmoid(x7)
        if refine_border:
            x = x[..., anchor[0]:anchor[1], anchor[2]:anchor[3]]

        return x

class Inpaint_Color_Net(nn.Module):
    def __init__(self, layer_size=7, upsampling_mode='nearest', add_hole_mask=False, add_two_layer=False, add_border=False):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        in_channels = 6
        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512+512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512+512, 512, activ='leaky')

        self.dec_5A = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4A = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3A = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2A = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1A = PCBActiv(64 + in_channels, 3, bn=False, activ=None, conv_bias=True)
        '''
        self.dec_5B = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4B = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3B = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2B = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1B = PCBActiv(64 + 4, 1, bn=False, activ=None, conv_bias=True)
        '''
    def cat(self, A, B):
        return torch.cat((A, B), dim=1)

    def upsample(self, feat, mask):
        feat = F.interpolate(feat, scale_factor=2, mode=self.upsampling_mode)
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        return feat, mask

    def forward_3P(self, mask, context, rgb, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((rgb, edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h) # + 128
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w) # + 256
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            # enlarge_input[:, 3] = 1. - enlarge_input[:, 3]
            enlarge_input = enlarge_input.to(cuda)
            rgb_output = self.forward(enlarge_input)
            rgb_output = rgb_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]

        return rgb_output

    def forward(self, input, add_border=False):
        input_mask = (input[:, -2:-1] + input[:, -1:]).clamp(0, 1)
        H, W = input.shape[-2:]
        f_0, h_0 = input, input_mask.repeat((1,input.shape[1],1,1))
        f_1, h_1 = self.enc_1(f_0, h_0)
        f_2, h_2 = self.enc_2(f_1, h_1)
        f_3, h_3 = self.enc_3(f_2, h_2)
        f_4, h_4 = self.enc_4(f_3, h_3)
        f_5, h_5 = self.enc_5(f_4, h_4)
        f_6, h_6 = self.enc_6(f_5, h_5)
        f_7, h_7 = self.enc_7(f_6, h_6)

        o_7, k_7 = self.upsample(f_7, h_7)
        o_6, k_6 = self.dec_7(self.cat(o_7, f_6), self.cat(k_7, h_6))
        o_6, k_6 = self.upsample(o_6, k_6)
        o_5, k_5 = self.dec_6(self.cat(o_6, f_5), self.cat(k_6, h_5))
        o_5, k_5 = self.upsample(o_5, k_5)
        o_5A, k_5A = o_5, k_5
        o_5B, k_5B = o_5, k_5
        ###############
        o_4A, k_4A = self.dec_5A(self.cat(o_5A, f_4), self.cat(k_5A, h_4))
        o_4A, k_4A = self.upsample(o_4A, k_4A)
        o_3A, k_3A = self.dec_4A(self.cat(o_4A, f_3), self.cat(k_4A, h_3))
        o_3A, k_3A = self.upsample(o_3A, k_3A)
        o_2A, k_2A = self.dec_3A(self.cat(o_3A, f_2), self.cat(k_3A, h_2))
        o_2A, k_2A = self.upsample(o_2A, k_2A)
        o_1A, k_1A = self.dec_2A(self.cat(o_2A, f_1), self.cat(k_2A, h_1))
        o_1A, k_1A = self.upsample(o_1A, k_1A)
        o_0A, k_0A = self.dec_1A(self.cat(o_1A, f_0), self.cat(k_1A, h_0))

        return torch.sigmoid(o_0A)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

class Discriminator(BaseNetwork):
    def __init__(self, use_sigmoid=True, use_spectral_norm=True, init_weights=True, in_channels=None):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not True), True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not True), True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
