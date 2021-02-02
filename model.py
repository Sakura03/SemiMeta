import torch, math
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

####################################################################
###################### ConvLarge Architecture ######################
####################################################################
class ConvLarge(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, stochastic=True, top_bn=False):
        super(ConvLarge, self).__init__()
        self.block1 = self.conv_block(input_dim, 128, 3, 1, 1, 0.1)
        self.block2 = self.conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = self.conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = self.conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = self.conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = self.conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = self.conv_block(256, 512, 3, 1, 0, 0.1)
        self.block8 = self.conv_block(512, 256, 1, 1, 0, 0.1)
        self.block9 = self.conv_block(256, 128, 1, 1, 0, 0.1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]
        if stochastic: maxpool.append(nn.Dropout2d())
        self.maxpool = nn.Sequential(*maxpool)
        
        classifier = [nn.Linear(128, num_classes)]
        if top_bn: classifier.append(nn.BatchNorm1d(num_classes))
        self.classifier = nn.Sequential(*classifier)

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
        return nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope)
                )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.maxpool(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)

        feature = self.avg_pool(out)
        feature = feature.view(feature.shape[0], -1)
        logits = self.classifier(feature)
        
        return logits

######################################################################
###################### Shake-Shake Architecture ######################
######################################################################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ShakeShakeBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShakeShakeBlock, self).__init__()
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab

class Shake(Function):
    @staticmethod
    def forward(ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate = inp1.new(inp1.size(0), 1, 1, 1)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @staticmethod
    def backward(ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate = grad_output.data.new(grad_output.size(0), 1, 1, 1).uniform_(0, 1)
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1. - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training

def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)

class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShiftConvDownsample, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(2*in_channels, out_channels, kernel_size=1, groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, num_classes=1000, downsample='basic'):
        super(ResNet32x32, self).__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(channels * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                        )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes, out_channels=planes)
            else: assert False

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def shakeshake26(**kwargs):
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model

########################################################################
############################ Wide ResNet-28 ############################
########################################################################
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if self.activate_before_residual:
            x = out

        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        
        if not self.equalInOut:
            x = self.convShortcut(x)

        return x + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wideresnet28(**kwargs):
    return WideResNet(depth=28, **kwargs)

if __name__ == '__main__':
    model = ConvLarge(input_dim=3)

    img = torch.randn(5, 3, 32, 32)
    logits = model(img)
    print(logits.shape)

    model = shakeshake26(num_classes=10)
    logits = model(img)
    print(logits.shape)
    
    model = wideresnet28(num_classes=10)
    logits = model(img)
    print(logits.shape)

