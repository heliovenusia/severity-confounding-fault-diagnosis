"""
Compact Temporal Convolutional Network for the isolation task.
One fixed architecture (only the input channel count, ny in
{2,4}, varies across conditions) -- no architecture search across
conditions; the architecture itself was chosen once, using TRAIN/VAL data
only on two pilot conditions (one low-A*, one high-A*), before being
frozen and reused identically for all 9 conditions and all 3 seeds.

Causal vs. non-causal: causal (left-only) padding is used. Justification:
the motivation here is a diagnostic method a real monitoring
system could run online, classifying based on data available up to and
including the current sample, not a method that is only well-defined once
an entire future window is already in hand. Using causal convolutions here
keeps the TCN consistent with that deployment framing, even though for
THIS SPECIFIC offline batch-classification task (fixed L=8 window,
classified as a whole after the fact) a non-causal/acausal variant would
have equal access to the same information and would very likely perform
about the same, since L=8 is small enough that the causal receptive field
already covers the entire window (see architecture parameters below).

Architecture (frozen, documented in full):
  - input: (batch, ny, L) with ny in {2, 4}, L=8
  - 3 residual TCN blocks, dilations (1, 2, 4), kernel_size=3
  - each block: [causal Conv1d -> ReLU -> Dropout] x2, + residual
    connection (1x1 Conv1d projection when in/out channel counts differ)
  - hidden_channels = 16 (fixed across all 3 blocks)
  - dropout = 0.1
  - global average pool over the time axis after the last block
  - linear head: Linear(16, 1) -> single logit (sigmoid applied outside,
    via BCEWithLogitsLoss during training and torch.sigmoid at inference)

Receptive field check: with kernel=3 and 2 conv layers per block, each
block contributes 2*(kernel-1)*dilation to the receptive field. Summed
over dilations (1,2,4): 2*2*1 + 2*2*2 + 2*2*4 = 4+8+16 = 28, comfortably
exceeding L=8 -- the last output timestep's receptive field already covers
the entire input window, so global average pooling over all timesteps
(rather than just reading the last one) is used to combine information
robustly rather than relying on the receptive field argument alone.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # left-pad only: causal
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    HIDDEN_CHANNELS = 16
    KERNEL_SIZE = 3
    DILATIONS = (1, 2, 4)
    DROPOUT = 0.1

    def __init__(self, ny: int):
        super().__init__()
        self.ny = ny
        channels = [ny] + [self.HIDDEN_CHANNELS] * len(self.DILATIONS)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(channels[i], channels[i + 1], self.KERNEL_SIZE,
                              self.DILATIONS[i], self.DROPOUT)
            for i in range(len(self.DILATIONS))
        ])
        self.head = nn.Linear(self.HIDDEN_CHANNELS, 1)

    def forward(self, x):
        # x: (batch, ny, L)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=2)  # global average pool over time -> (batch, hidden)
        return self.head(pooled).squeeze(-1)  # (batch,) raw logit

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
