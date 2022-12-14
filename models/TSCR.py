# ENCODE POSITION WITH SG
# ADD TO DESCRIPTOR
# PASS TO ENCODER
# PASS TO DECODER QUERIES ARE ENCODED SCENE ID ENCIDED POSITION
# APPLY REGGRESSION HEAD OF 3D COORDS
# RETURB 2D-3D MATCHES
# AT INFERENCE TINE APPLY RANSAC

import torch.nn as nn
import torch
import torch.nn.functional as F
#https://github.com/Tangshitao/Dense-Scene-Matching/blob/3bdf349dd4b9e34ba8b4ac6669ff0b8e8cd40d36/libs/model/arch/DSMNet.py

## Normalization method from SuperGlue
def normalize_keypoints(kpts, img_shape):
    H, W = img_shape[0]
    h = H - 1
    w = W - 1
    norm_kpts = torch.zeros_like(kpts)
    norm_kpts[:, :, 0] = (kpts[:, :, 0] - w / 2) / (w / 2)
    norm_kpts[:, :, 1] = (kpts[:, :, 1] - h / 2) / (h / 2)
    return norm_kpts
'''
def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    if image_shape.shape[0] == 1: # batch size is 1
        height, width = image_shape[0]
        one = kpts.new_tensor(1)
        size = torch.stack([one*width, one*height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.7
        return (kpts - center[:, None, :]) / scaling[:, None, :]
    else:
        norm_kpts = torch.zeros(kpts.shape).to(kpts.device).to(kpts.dtype)
        for i in range(image_shape.shape[0]):
            height, width = image_shape[i]
            one = kpts[i].new_tensor(1)
            size = torch.stack([one * width, one * height])[None]
            center = size / 2
            scaling = size.max(1, keepdim=True).values * 0.7
            norm_kpts[i] = (kpts[i] - center[:, None, :]) / scaling[:, None, :]
        return norm_kpts
'''
## MLP method from SuperGlue
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class XYZregressor(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.fc_o = nn.Linear(ch, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        x = F.gelu(self.fc_h(x))
        return self.fc_o(x)


class TSCR(nn.Module):

    def __init__(self, config):
        super().__init__()
        desc_dim = config.get("desc_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        self.multiscene = config.get("multiscene")


        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                               nhead=nhead,
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout,
                                                               activation='gelu', batch_first=True)

        #self.kp_encoder = MLP([2, 32, 64, 128, 256, desc_dim])
        self.kp_encoder = nn.Sequential(nn.Linear(2, desc_dim // 2), nn.GELU(),
                                           nn.Linear(desc_dim // 2, desc_dim), nn.GELU(),
                                           nn.Linear(desc_dim, d_model))

        if self.multiscene:
            self.scene_encoder = MLP([1,  32, 64, 128, 256, d_model])

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                             num_layers=config.get("num_encoder_layers"),
                                                             norm=nn.LayerNorm(d_model))

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu',
                                                               batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))

        self.xyz_regressor = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                           nn.Linear(d_model * 2, d_model), nn.GELU(),
                                           nn.Linear(d_model, 3))

        num_kps = config.get("n_kps")
        self.query_embed = nn.Parameter(torch.zeros((num_kps, d_model)), requires_grad=True)
        #nn.init.uniform_(self.query_embed.weight)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, data):
        kps = normalize_keypoints(data["keypoints"], data["shape"])
        batch_size = kps.shape[0]
        encoded_kps = self.kp_encoder(kps)
        features = data["descriptors"] + encoded_kps

        enc_features = self.transformer_encoder(features)
        latent_xyz = self.transformer_decoder(self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1), enc_features)
        if self.multiscene:
            encoded_scene = self.scene_encoder(self.data["scene"])
            latent_xyz = latent_xyz + encoded_scene
        xyz = self.xyz_regressor(latent_xyz)

        return xyz
