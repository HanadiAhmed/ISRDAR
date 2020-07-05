from model import common
import torch.nn as nn
import torch.nn.functional as F
import torch
##########  Name of model #################
def make_model(args, parent=False):
    return ISR(args)
################################################
# Feature layer
class Feat_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True):
        super(Feat_Layer, self).__init__()
        self.feat_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=(kernel_size//2), bias=bias)
        )

    def forward(self, x):
        y = self.feat_1(x)
        return y
###########################################
###################################################################
class Space_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Space_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        # downscale = scale + 4

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        #self.bn = nn.BatchNorm2d(output_size)
        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)


    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(x)
        # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        # if self.type == 'softmax':
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W
        #output = self.bn(output)
        return output
########################################################################
#################################################################################################################################################################
## Channel and Spacial Attentions-Residual-Dense Networks
class ISR(nn.Module):
    def __init__(self, args, conv=Feat_Layer):
        super(ISR, self).__init__()

        n_Blk = args.n_Blk
        n_resgroups = args.n_resgroups
        reduction = args.reduction

        n_feats = args.n_feats
        kernel_size = 3

        scale = args.scale[0]
        act = nn.ReLU(True)
        ##########################################################################################################
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        #########################################################################################################
        # define head module
        h=[nn.Conv2d(n_feats*5, n_feats, kernel_size)]
        self.he = nn.Sequential(*h)

        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
       ############################################################################################################

        ###############################################################################################################

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        ###############################################################################################################
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        ##############################################################################################################
        self.head = nn.Sequential(*modules_head)
        ###############################################
        self.SA0 = Space_attention(n_feats, n_feats, 1, 1, 0, 1)
        #################################################3

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)


        self.ac = torch.nn.PReLU()
        ########################################################################
        ##############################################

        modules_convo1 = [conv(n_feats, n_feats, kernel_size)]

        self.convo1 = nn.Sequential(*modules_convo1)

        ###############################################################################################################
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
       # x1 = self.ac(x)
       # x1 = self.convo1(x1)
        x1 = self.SA0(x)
        xx = self.tail(x1)
        xx = self.add_mean(xx)

        return xx
##############################################################################################
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))