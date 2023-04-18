import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

from utils.utils import GPU


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x * (torch.tanh(F.softplus(x)))


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.sin(x)


class Swish(nn.Module):
    r"""Applies the element-wise function :math:`\text{Swish}(x) = x * \frac{1}{1 + \exp(-x)}`

        Shape:
            - Input: :math:`(N, *)` where `*` means, any number of additional
              dimensions
            - Output: :math:`(N, *)`, same shape as the x

        .. image:: scripts/activation_images/Sigmoid.png

        Examples::

            >>> m = Swish()
            >>> x = torch.randn(2)
            >>> output = m(x)
        """
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, **kwargs):
        result = x*x.sigmoid()
        ctx.save_for_backward(result, x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, x = ctx.saved_variables
        sigmoid_x = x.sigmoid()
        return grad_output * (result+sigmoid_x*(1-result))


def distance_loss(output, target, weight=0.1, distance_metric=2):
    dist = (torch.dist(output, target, p=distance_metric) / output.shape[0]) * weight
    return dist


def contrastive_distance_loss(output_batch, target_batch, weight=0.1, distance_metric=2, self_weight=1, other_weight=1):
    assert output_batch.shape == target_batch.shape, 'Output and target must have the same shape! {} != {}'\
        .format(output_batch.shape, target_batch.shape)
    batch_size = len(output_batch)
    distance_self = torch.zeros(1).to(output_batch.device)
    distance_other = torch.zeros(1).to(output_batch.device)
    for i in range(batch_size):
        for n in range(batch_size):
            if i == n:
                distance_self -= torch.dist(output_batch[i], target_batch[n], p=distance_metric)
            else:
                distance_other += torch.dist(output_batch[i], target_batch[n], p=distance_metric)

    dist = ((distance_self * self_weight + distance_other * other_weight) / output_batch.shape[0]) * weight
    return dist.squeeze_(0)


class DistLoss(torch.nn.Module):
    def __init__(self, weight=0.1, distance_metric=2, self_weight=1, other_weight=1):
        super(DistLoss, self).__init__()
        self.weight = weight
        self.self_weight = self_weight
        self.other_weight = other_weight
        self.distance_metric = distance_metric

    def forward(self, output, target):
        return contrastive_distance_loss(output, target, self.weight, self.distance_metric, self.self_weight, self.other_weight)


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class GMSDLoss(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()

        self.hx = nn.Conv2d(in_channel, 3, (3, 3), stride=1, padding=1, bias=False, groups=in_channel)
        self.hx.weight.data = torch.tensor([[1 / 3.0, 0, -1 / 3.0],
                                            [1 / 3.0, 0, -1 / 3.0],
                                            [1 / 3.0, 0, -1 / 3.0]]).expand([in_channel, 1, 3, 3])
        self.hx.weight.requires_grad = False

        self.hy = nn.Conv2d(in_channel, 3, (3, 3), stride=1, padding=1, bias=False, groups=in_channel)
        self.hy.weight.data = torch.tensor([[1 / 3.0, 1 / 3.0, 1 / 3.0],
                                            [0, 0, 0],
                                            [-1 / 3.0, -1 / 3.0, -1 / 3.0]]).expand([in_channel, 1, 3, 3])
        self.hy.weight.requires_grad = False

        self.eps = 1e-9

    def forward(self, pred, target):
        H, W = pred.shape[-2], pred.shape[-1]
        mr = torch.sqrt(self.hx(target) ** 2 + self.hy(target) ** 2)
        md = torch.sqrt(self.hx(pred) ** 2 + self.hy(pred) ** 2)

        gms = (2 * mr * md + self.eps) / (mr ** 2 + md ** 2 + self.eps)
        gmsd = torch.std(gms, dim=(-2, -1))

        return gmsd.mean()


def init_weights(modules):
    pass
    # for m in modules:
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def shape_description(shape, start=1, stop=None, reverse=True):
    shape = list(shape)
    shape = shape[start:] if stop is None else shape[start:stop]
    if reverse:
        shape.reverse()
    format_string = 'x'.join(['{}' for _ in shape])
    return format_string.format(*shape)


def log_shapes(name, prev_shape, cur_shape):
    shape_desc1 = shape_description(prev_shape)
    shape_desc2 = shape_description(cur_shape)
    logging.info('{}:\t{:13} -> {:13}'.format(name, shape_desc1, shape_desc2))


def reparameterization(mu, logvar, latent_size=100):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_size))).to(GPU.device))
    z = sampled_z * std + mu
    return z


def sample(mu, logvar):
    std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(GPU.device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class GuidedBackprop():
    """
        Visualize CNN activation maps with guided backprop.

        Returns: An image that represent what the network learnt for recognizing
        the given image.

        Methods: First layer input that minimize the error between the last layers output,
        for the given class, and the true label(=1).

        ! Call visualize(image) to get the image representation
    """

    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        # eval mode
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_out, grad_in):
            """ Return reconstructed activation image"""
            self.image_reconstruction = grad_out[0]

        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_out, grad_in):
            """ Output the grad of model output wrt. layer (only positive) """

            # Gradient of forward_output wrt. forward_input = error of activation map:
            # for relu layer: grad of zero = 0, grad of identity = 1
            grad = self.activation_maps[-1]  # corresponding forward pass output
            grad[grad > 0] = 1  # grad of relu when > 0

            # set negative output gradient to 0 #!???
            positive_grad_out = torch.clamp(input=grad_out[0], min=0.0)

            # backward grad_out = grad_out * (grad of forward output wrt. forward input)
            new_grad_out = positive_grad_out * grad

            del self.forward_outputs[-1]

            # For hook functions, the returned value will be the new grad_out
            return (new_grad_out,)

        # !!!!!!!!!!!!!!!! change the modules !!!!!!!!!!!!!!!!!!
        # only conv layers, no flattened fc linear layers
        modules = list(self.model.features._modules.items())

        # register hooks to relu layers
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        # register hook to the first layer
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class):
        # last layer output
        model_output = self.model(input_image)
        self.model.zero_grad()

        # only calculate gradients wrt. target class
        # set the other classes to 0: eg. [0,0,1]
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        grad_target_map[0][target_class] = 1

        model_output.backward(grad_target_map)

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        result = self.image_reconstruction.data.numpy()[0]
        return result