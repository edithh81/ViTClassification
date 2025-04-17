import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import cv2
import os

class TransfomerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransfomerEncoder, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim, bias=True)
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        # norm_1 = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.dropout1(attn_output)
        out_1 = self.norm1(x+attn_output) # skip connection
        ffn_output = self.ff(out_1)
        ffn_output = self.dropout2(ffn_output)
        out_2 = self.norm2(out_1+ffn_output)
        return out_2
class PatchPositionEmbedding(nn.Module): # add CLS token
    def __init__(self, embed_dim, patch_size, img_size):
        super(PatchPositionEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = 1.0 / np.sqrt(embed_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(scale*torch.randn((img_size // patch_size) ** 2  , embed_dim))
    def forward(self, x):
        x = self.conv1(x) # shape: (batch_size, embed_dim, h, w)
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape: (batch_size, embed_dim, num_patches)
        x = x.permute(0, 2, 1) # permute to (batch_size, num_patches, embed_dim)
        # cls_embed = self.cls_token.to(x.dtype) + torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device) # shape: (batch_size, 1, embed_dim)
        # x = torch.cat((cls_embed, x), dim=1) # shape: (batch_size, num_patches + 1, embed_dim)
        x = x + self.position_embedding.to(x.dtype) # shape: (batch_size, num_patches + 1, embed_dim)
        return x

class ViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, num_classes, patch_size, drop_out, img_size=224):
        super(ViT, self).__init__()
        self.patch_embedding = PatchPositionEmbedding(embed_dim, patch_size, img_size=img_size)
        self.transformer_encoder = TransfomerEncoder(embed_dim, num_heads, num_layers, ff_dim, drop_out)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(drop_out)
    def forward(self, x):
        # take position embedding
        output = self.patch_embedding(x)
        output = self.transformer_encoder(output)
        output = output.mean(dim=1) # global average pooling
        output = self.fc1(output)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

        
        