import torch.nn as nn
import torch.nn.functional as F

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer (transposed convolution) for simplicity.
    
    Args:
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        k_size (int): Kernel size
        stride (int): Stride for the convolution
        pad (int): Padding size
        bn (bool): Whether to include batch normalization
        
    Returns:
        nn.Sequential: Sequential container of layers
    """
    layers = [nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity.
    
    Args:
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        k_size (int): Kernel size
        stride (int): Stride for the convolution
        pad (int): Padding size
        bn (bool): Whether to include batch normalization
        
    Returns:
        nn.Sequential: Sequential container of layers
    """
    layers = [nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):
    """Generator for transferring from MNIST to SVHN (Domain 1 to Domain 2).
    
    Architecture:
        - Two encoding convolutional blocks
        - Two residual blocks
        - Two decoding transposed convolutional blocks
        
    Args:
        conv_dim (int): Base number of convolutional filters (default: 64)
    """

    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks (downsampling)
        self.conv1 = conv(1, conv_dim, 4)  # 1 channel input (MNIST grayscale)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks (maintain same dimension)
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks (upsampling)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)  # 3 channels output (SVHN RGB)

    def forward(self, x):
        """Forward pass of the generator.
        
        Args:
            x (torch.Tensor): Input tensor (MNIST image)
            
        Returns:
            torch.Tensor: Generated output (SVHN-like image)
        """
        # Encoding path
        out = F.leaky_relu(self.conv1(x), 0.05)  # (batch_size, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (batch_size, 128, 8, 8)

        # Residual blocks
        out = F.leaky_relu(self.conv3(out), 0.05)  # (batch_size, 128, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (batch_size, 128, 8, 8)

        # Decoding path
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (batch_size, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (batch_size, 3, 32, 32)
        return out


class G21(nn.Module):
    """Generator for transferring from SVHN to MNIST (Domain 2 to Domain 1).
    
    Architecture similar to G12 but with reversed channel dimensions.
    """

    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        # 32x32x3 input (SVHN RGB)
        self.conv1 = conv(3, conv_dim, 4)  # 3 channel input (SVHN RGB)
        # 16x16x64 output
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)  # 1 channel output (MNIST grayscale)

    def forward(self, x):
        """Forward pass of the generator.
        
        Args:
            x (torch.Tensor): Input tensor (SVHN image)
            
        Returns:
            torch.Tensor: Generated output (MNIST-like image)
        """
        out = F.leaky_relu(self.conv1(x), 0.05)  # (batch_size, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (batch_size, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)  # (batch_size, 128, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (batch_size, 128, 8, 8)

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (batch_size, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (batch_size, 1, 32, 32)
        return out


class D1(nn.Module):
    """Discriminator for MNIST images.
    
    Args:
        conv_dim (int): Base number of convolutional filters (default: 64)
        use_labels (bool): Whether to output class labels (for conditional GAN)
    """

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)  # No BN on first layer
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        
        # Output 11 channels if using labels (10 digits + fake class), else 1
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 3, 1, 0, False)  # Final classification layer

    def forward(self, x):
        """Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor (real or generated image)
            
        Returns:
            torch.Tensor: Discriminator output (logits)
        """
        out = F.leaky_relu(self.conv1(x), 0.05)  # (batch_size, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (batch_size, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (batch_size, 256, 4, 4)
        
        out = self.fc(out).squeeze()  # Remove unnecessary dimensions
        return out


class D2(nn.Module):
    """Discriminator for SVHN images.
    
    Similar architecture to D1 but accepts 3-channel input.
    """

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)  # 3 channel input (SVHN RGB)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 3, 1, 0, False)

    def forward(self, x):
        """Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor (real or generated image)
            
        Returns:
            torch.Tensor: Discriminator output (logits)
        """
        out = F.leaky_relu(self.conv1(x), 0.05)  # (batch_size, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (batch_size, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (batch_size, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out
    

"""
The code appears to be part of a CycleGAN implementation for domain transfer between MNIST and SVHN datasets, with:

G12 converting MNIST to SVHN-like images

G21 converting SVHN to MNIST-like images

D1 discriminating real MNIST from generated MNIST

D2 discriminating real SVHN from generated SVHN
"""