from typing import Union

import torch.nn as nn
from monai.networks.layers import Conv, Pool


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        if self.grid.device != flow.device:
            self.grid = self.grid.to(flow.device)

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DeformableConvLayer(nn.Module):

    def __init__(self, img_size, in_channel, out_channel, kernel_size, stride, spatial_dims=2, max_view=16, sample_rate=0.1, dfc_rate=1.0, in_channel_group=4,
                 in_channel_fold=True):
        super().__init__()
        assert 0 <= dfc_rate <= dfc_rate * out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_view = max_view
        self.dfc_rate = dfc_rate
        self.in_channel_fold = in_channel_fold
        self.in_channel_group = in_channel_group

        if in_channel_fold:
            self.ic_fd = Conv[Conv.CONV, spatial_dims](
                in_channels=self.in_channel, out_channels=1, kernel_size=1, stride=1, padding=0
            )
            self.in_channel = 1

        self.group_size = max(int((max_view ** 2) * sample_rate), 2)

        self.fusion = Conv[Conv.CONV, spatial_dims](
            in_channels=self.group_size, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.conv = Conv[Conv.CONV, spatial_dims](
            in_channels=self.in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )

        self.offset_cov = Conv[Conv.CONV, spatial_dims](
            in_channels=self.in_channel, out_channels=int(out_channel * dfc_rate) * spatial_dims * self.group_size, kernel_size=1, stride=1, padding=0
        )
        self.offset_embedding = Pool[Pool.ADAPTIVEAVG, spatial_dims](
            output_size=1
        )

        self.one_grid = torch.ones((spatial_dims,) + img_size)
        self.trans = SpatialTransformer(img_size)

    def forward(self, x):
        assert x.ndim in [4, 5]
        ndim = x.ndim - 2
        if ndim == 2:
            B, C, _, _ = x.shape
        elif ndim == 3:
            B, C, _, _, _ = x.shape

        if self.in_channel_fold:
            x = self.ic_fd(x)

        dfc_channel = int(self.out_channel * self.dfc_rate)

        offset_embed = self.offset_embedding(self.offset_cov(x)).view((B, self.group_size, dfc_channel, ndim))

        x_f = torch.repeat_interleave(x, self.group_size, dim=1)
        scale_ = self.max_view / 2
        # from -1/2 * self.max_view to 1 / 2 * self.max_view
        normal_offset = torch.sigmoid(offset_embed * scale_)
        max_offset = normal_offset.max().detach()
        normal_offset = normal_offset / (max_offset + 1e-5)
        offset = normal_offset * self.max_view - self.max_view / 2
        # keep the center should at the  slid window middle position
        offset -= offset.mean().detach()
        x_f_list = [x_f[:, i::self.in_channel] for i in range(self.in_channel)]
        one_grid = torch.stack([self.one_grid] * B * self.group_size)

        x_rerange = torch.concat(x_f_list, dim=1)

        out = 0
        _, _, H, W = x_rerange.shape  # (B, group_size * C_in, H, W)

        offset_embed_batched = offset_embed.reshape(B * self.group_size, dfc_channel, ndim)  # (B * group_size, C_out, H, W)
        f_batched = x_rerange.reshape(B * self.group_size, self.in_channel, H, W)  # (B * group_size, C_in, H, W)

        for oc in range(dfc_channel):
            f = self.trans(f_batched, one_grid * offset_embed_batched[:, oc, :][..., None, None])
            f = self.fusion(f.view((B, self.group_size, self.in_channel, H, W))
                            .view(B, self.group_size, self.in_channel * H, W)) \
                .view((B, self.in_channel, H, W))

            out += f

        return self.conv(out + x)


class DSConv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding: Union[str, int, tuple] = 0,
                 dilation=1
                 ):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class HugeConv2dBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=(31, 31), small_ks=5):
        super().__init__()
        self.w, self.h = kernel_size
        self.conv0 = DSConv2d(in_channel, out_channel, (self.w, small_ks), padding="same")
        self.conv1 = DSConv2d(in_channel, out_channel, (small_ks, self.h), padding="same")
        self.sconv = nn.Conv2d(in_channel, out_channel, (small_ks, small_ks), padding="same")

    def forward(self, x):
        main_path = self.conv0(x)
        main_path = main_path + self.conv1(x)
        path = self.sconv(x)

        return path + main_path


######################## SubConv ########################

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel_size=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel_size, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x


#################### Involution-CUDA #########################

from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_involution_kernel = kernel_loop + '''
extern "C"
__global__ void involution_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_involution_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_involution_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class _involution(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('involution_forward_kernel', _involution_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_input_kernel',
                                _involution_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_weight_kernel',
                                _involution_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _involution_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ involution kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _involution.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class Involution(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=1),
                nn.BatchNorm2d(channels // reduction_ratio),
                nn.ReLU()
            ]
        )

        self.conv2 = nn.Conv2d(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size - 1) // 2)
        return out


###################### Involution Native ######################################


class InvolutionNative(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(InvolutionNative, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=1),
                nn.BatchNorm2d(channels // reduction_ratio),
                nn.ReLU()
            ]
        )

        self.conv2 = nn.Conv2d(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
