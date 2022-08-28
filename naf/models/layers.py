import torch.nn as nn
import torch
from monai.networks.layers import DropPath, trunc_normal_, Conv, Pool, get_act_layer, get_norm_layer


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

    def __init__(self, img_size, in_channel, out_channel, kernel_size, stride, spatial_dims=2, max_view=16, sample_rate=0.1, dfc_rate=1.0, in_channel_group=4, in_channel_fold=True):
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