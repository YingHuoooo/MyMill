import torch
import torch.nn
from typing import Dict
import ocnn
from ocnn.octree import Octree


def expand_batch_features(features: torch.Tensor, counts) -> torch.Tensor:
    expanded = []
    for i in range(features.size(0)):
        count = int(counts[i])
        expanded.append(features[i].unsqueeze(0).expand(count, -1))
    return torch.cat(expanded, dim=0)


class ToolConditionMLP(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, tool_params: torch.Tensor) -> torch.Tensor:
        return self.net(tool_params)

class UNet(torch.nn.Module):
    r''' Octree-based UNet for segmentation.
    '''

    def __init__(self, in_channels: int, out_channels: int, interp: str = 'linear',
                 nempty: bool = False, conditioning: str = 'concat', **kwargs):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nempty = nempty
        self.conditioning = conditioning.lower()
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)
        self.batch_size = 1

        # encoder
        self.conv1 = ocnn.modules.OctreeConvBnRelu(
            in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

        # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                   for i in range(self.decoder_stages)]
        channel = [c + 256 for c in channel]
        if self.conditioning == 'film':
            condition_channels = [2 * c for c in self.decoder_channel[1:]]
        elif self.conditioning != 'concat':
            raise ValueError('Unsupported conditioning: %s' % conditioning)

        self.upsample = torch.nn.ModuleList([ocnn.modules.OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])

        # header
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)
        self.header = torch.nn.Sequential(
            ocnn.modules.Conv1x1BnRelu(self.decoder_channel[-1], self.head_channel),
            ocnn.modules.Conv1x1(self.head_channel, self.out_channels, use_bias=True))
        self.header_2 = torch.nn.Sequential(
            ocnn.modules.Conv1x1BnRelu(self.decoder_channel[-1], self.head_channel),
            ocnn.modules.Conv1x1(self.head_channel, self.out_channels, use_bias=True))

        self.fc_module_1 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )

        self.fc_module_2 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )
        self.fc_module_3 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )
        self.fc_module_4 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )

        if self.conditioning == 'film':
            self.film_conditioners = torch.nn.ModuleList([
                ToolConditionMLP(c) for c in condition_channels
            ])

    def config_network(self):
        r''' Configure the network channels and Resblock numbers.
        '''
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.decoder_channel = [256, 256, 128, 96, 96]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2, 2, 2]
        self.head_channel = 64
        self.bottleneck = 1
        self.resblk = ocnn.modules.OctreeResBlock2

    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):
        r''' The encoder of the U-Net.
        '''
        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree,
                     depth: int, tool_features, film_conditions=None):
        r''' The decoder of the U-Net.
        '''
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)

            copy_counts = octree.batch_nnum[i + 2]
            if film_conditions is not None:
                expanded_film = expand_batch_features(film_conditions[i], copy_counts)
                gamma, beta = torch.chunk(expanded_film, 2, dim=1)
                deconv = deconv * (1.0 + gamma) + beta

            expanded_tool_features = expand_batch_features(tool_features[i], copy_counts)
            deconv = torch.cat([expanded_tool_features, deconv], dim=1)

            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int,
                query_pts: torch.Tensor, tool_params: torch.Tensor):
        r''' Forward pass with tool parameters incorporated.
        '''

        convd = self.unet_encoder(data, octree, depth)

        tool_features = [
            self.fc_module_1(tool_params),
            self.fc_module_2(tool_params),
            self.fc_module_3(tool_params),
            self.fc_module_4(tool_params),
        ]
        film_conditions = None
        if self.conditioning == 'film':
            film_conditions = [conditioner(tool_params)
                               for conditioner in self.film_conditioners]

        deconv = self.unet_decoder(
            convd, octree, depth - self.encoder_stages, tool_features,
            film_conditions)

        interp_depth = depth - self.encoder_stages + self.decoder_stages
        # print(f"deconv shape: {deconv.shape}")
        # print(f"octree depth: {interp_depth}, query_pts shape: {query_pts.shape}")
        feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        # print(f"query_pts shape: {query_pts.shape}")
        # print(f"deconv batch size: {deconv.shape[0]}")
        # print(f"octree batch size: {octree.batch_size}")
        # print(f"query_pts batch size: {query_pts.shape[0]}")
        logits_1 = self.header(feature)
        logits_2 = self.header_2(feature)
        return logits_1,logits_2
