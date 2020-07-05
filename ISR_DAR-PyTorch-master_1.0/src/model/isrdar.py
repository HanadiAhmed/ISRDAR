from model import common
import torch.nn as nn
import torch.nn.functional as F
import torch
##########  Name of model #################
def make_model(args, parent=False):
    return ISRDAR(args)
################################################
# Feature layer
class Feat_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True):
        super(Feat_Layer, self).__init__()
        self.feat_1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=(kernel_size//2), bias=bias)


    def forward(self, x):
        y = self.feat_1(x)
        return y
###########################################
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, in_channel,out_channel,reduction):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.conv_1 = nn.Conv2d(in_channel,out_channel//reduction , 1, padding=0, bias=True)
        self.relu_1= nn.PReLU()
        self.conv_2=  nn.Conv2d(out_channel//reduction ,out_channel, 1, padding=0, bias=True)
        self.seg_1=   nn.Sigmoid()


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_1(y)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.seg_1(y)
        return x * y
###############################################################

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,reduction,
            bias=True, bn=False, act=nn.ReLU(True), scale=1):

        super(CAB, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.MAX_pool = nn.MaxPool2d(1)
        self.Se = nn.Sigmoid()
        modules_body = []
        for i in range(2):
            if i == 1: modules_body.append(CALayer(n_feat, n_feat,reduction))
            else:
             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
             if i == 0: modules_body.append(act)

        modules_body.append(CALayer(n_feat,n_feat,reduction))
        self.body = nn.Sequential(*modules_body)
        self.scale = scale

    def forward(self, x):
       # y = self.avg_pool(x)
        v=self.MAX_pool(x)
        #y1=y+v
        ch = self.body(v)
        z = self.Se(ch)
        return z*x
##########################################################################
## Special Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self, in_channel,out_channel,reduction):
        super(SALayer, self).__init__()

        # feature channel downscale and upscale --> channel weight
        self.conv_= nn.Sequential(
            nn.Conv2d(in_channel,out_channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//reduction , out_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        y = self.conv_(x)
        return x * y

###################################################
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
class Time_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Time_attention, self).__init__()

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
        self.up= common.Upsampler(output_size, scale, output_size, act=False)
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
        y=self.up(x)
        V = self.V(y)
        #V = self.V(y)
        # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        #attention = x
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
        output = W
        #output = self.bn(output)
        return output
    ####################

################################################
##  Special Attention Block (SAB)
class SAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,reduction,
            bias=True, bn=False, act=nn.ReLU(True), scale=1):

        super(SAB, self).__init__()
        self.Se = nn.Sigmoid()
        modules_body = []
        for i in range(2):
             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
             if i == 0: modules_body.append(act)
        modules_body.append(SALayer(n_feat,n_feat,reduction))
        self.body = nn.Sequential(*modules_body)
        self.scale = scale

    def forward(self, x):
        sp = self.body(x)
        z = self.Se(sp)
        return z*x
###################################################################
## Residual Layer
class ResLayer(nn.Module):
    def __init__(self,in_channel,out_channel,reduction):
        super(ResLayer, self).__init__()

        # feature channel downscale and upscale --> channel weight
        self.conv_= nn.Sequential(
            nn.Conv2d(in_channel,out_channel//reduction, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//reduction , out_channel, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=True)
        )

    def forward(self, x):

        y = self.conv_(x)
        return x + y

###################################################
## Residual  Block (RB)
class RB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,reduction,
            bias=True, bn=False, act=nn.ReLU(True), scale=1):

        super(RB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ResLayer(n_feat,n_feat,reduction))
        self.body = nn.Sequential(*modules_body)
        self.scale = scale

    def forward(self, x):
        res = self.body(x).mul(self.scale)
       # res =self.body(x)
        return res+x

############################################################################################################################
class Dense_Block(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size):
       super(Dense_Block, self).__init__()

       self.relu = nn.ReLU(inplace=True)
       self.bn = nn.BatchNorm2d(in_channels)

       self.conv1 = nn.Conv2d(in_channels,out_channels,  kernel_size, stride=1, padding=1)
       self.conv2 = nn.Conv2d(out_channels, out_channels,  kernel_size, stride=1, padding=1)
       self.conv3 = nn.Conv2d(out_channels*2, out_channels,  kernel_size, stride=1, padding=1)
       self.conv4 = nn.Conv2d(out_channels*3, out_channels,  kernel_size, stride=1, padding=1)
       self.conv5 = nn.Conv2d(out_channels*4, out_channels,  kernel_size, stride=1, padding=1)


    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
    # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))

        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))

        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))

        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense
"""
def forward(self, x):
    bn = self.bn(self.relu(self.conv(x)))
    out = self.avg_pool(bn)
    return out
"""

class DenseNet(nn.Module):
    def __init__(self, conv, out_channels, kernel_size):
        super(DenseNet, self).__init__()

        # Make Dense Blocks
        self.denseblock1 = self._make_dense_block(Dense_Block, out_channels,out_channels, kernel_size)
        #self.bn = nn.BatchNorm2d(out_channels)


        h = [nn.Conv2d(out_channels * 5, out_channels,1)]
       # h = [conv(out_channels * 2, out_channels,kernel_size)]
        self.he = nn.Sequential(*h)


    def _make_dense_block(self, block, in_channels,out_channels, kernel_size):
        layers = []
        layers.append(block(in_channels,out_channels, kernel_size))
        return nn.Sequential(*layers)


    def forward(self, x):
       # out = self.relu(x)
        out = self.denseblock1(x)
        out= self.he(out)
        return out
#####################################################################
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,reduction,
            bias=False, bn=True, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, n_feat,reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res
#############################

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,reduction,  act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

#################################################################################################################################################################
## Channel and Spacial Attentions-Residual-Dense Networks
class ISRDAR(nn.Module):
    def __init__(self, args, conv=Feat_Layer):
        super(ISRDAR, self).__init__()

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
        # define body module
        modules_body = [
            CAB(
                conv, n_feats, kernel_size,reduction,  act=act, scale=args.res_scale)\
            for _ in range(n_Blk)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        #################################
        modules_body1 = [
            SAB(
                conv, n_feats, kernel_size,reduction,  act=act, scale=args.res_scale) \
            for _ in range(n_Blk)]

        modules_body1.append(conv(n_feats, n_feats, kernel_size))

        #################################
        modules_body2 = [
            RB(
                conv, n_feats, kernel_size,reduction, act=act, scale=args.res_scale) \
            for _ in range(n_Blk)]

        modules_body2.append(conv(n_feats, n_feats, kernel_size))
        #################################
        modules_body3 = [
            DenseNet( conv, n_feats, kernel_size) \
            for _ in range(n_Blk)]

        modules_body3.append(conv(n_feats, n_feats, kernel_size))

        ###############################################################################################################

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        ###############################################################################################################
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        ##############################################################################################################
        self.head = nn.Sequential(*modules_head)

        self.body = nn.Sequential(*modules_body)
        self.body1 = nn.Sequential(*modules_body1)
        self.body2 = nn.Sequential(*modules_body2)
        self.body3 = nn.Sequential(*modules_body3)

        self.tail = nn.Sequential(*modules_tail)
         ##########################
        modules_body4 = [
            ResidualGroup(
                conv, n_feats, kernel_size,reduction,  act=act, res_scale=args.res_scale, n_resblocks=n_Blk) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body4 = nn.Sequential(*modules_body4)
        ##################################################################
        self.SA0 = Space_attention(n_feats, n_feats, 1, 1, 0, 1)
        # self.SA1 = Time_attention(n_feats, n_feats, 1, 1, 0, 1)
        modules_body32 = [
            Time_attention(n_feats, n_feats, 1, 1, 0, 1) \
            for _ in range(n_Blk)]

        #  modules_body32.append(conv(n_feats, n_feats, kernel_size))
        self.body32 = nn.Sequential(*modules_body32)
        self.ac = torch.nn.PReLU()
        ########################################################################
        ##############################################

        modules_convo1 = [conv(n_feats, n_feats, kernel_size)]

        self.convo1 = nn.Sequential(*modules_convo1)

        ###############################################################################################################
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        x1 = self.ac(x)
        x1 = self.convo1(x1)
        x1=self.SA0(x1)

        #######################
        CA = self.body(x1)
        SA = self.body32(x1)
        RES = self.body2(x1)
        DEN = self.body3(x1)
        ########################
        SUM = ( CA*DEN) + (SA * RES) # SUM +=x (good 2)
       # SUM =  (DEN*SA)+(CA * RES)  # SUM +=x (good 4)
        SUM = self.body4(SUM )
      #  SUM1 = self.body4(x)  #SUM += (x + CA + SA + DEN + RES + SUM1) #(good 2)

        SUM +=(x+CA+SA+DEN+RES) # The best (Num 1)

        xx = self.tail(SUM)
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
