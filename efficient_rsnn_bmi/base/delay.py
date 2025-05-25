from stork.connections import BaseConnection
from DCLS.construct.modules import Dcls1d
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from collections import deque

class CustomDelayConnection(BaseConnection):
    def __init__(
        self,
        src,
        dst,
        kernel_count,
        dilated_kernel_size,
        left_padding,
        right_padding,
        version="gauss",
        stride = 1,
        groups = 1,
        target=None,
        bias=False,
        requires_grad=True,
        propagate_gradients=True,
        name=None,
        regularizers=None,
        constraints=None,
        **kwargs
    ):
        super().__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.kernel_count = kernel_count
        self.groups = groups
        self.dilated_kernel_size = dilated_kernel_size
        self.version = version
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.stride = stride
        self.delay_buffer = deque()
        self.conv_out_size = None

        self.op = Dcls1d(
            in_channels=src.shape[0], 
            out_channels=dst.shape[0], 
            kernel_count=kernel_count,
            groups=groups,
            stride=stride,
            dilated_kernel_size=dilated_kernel_size,
            bias=bias, 
            version=version
        )

        self.conv_out_size = ((1 - self.dilated_kernel_size + (self.left_padding + self.right_padding)) / self.stride) + 1
        if not self.conv_out_size.is_integer() or self.conv_out_size <= 0:
            raise ValueError(f"Convolution Output Size must be positive integer, got {self.conv_out_size}")
        
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def add_diagonal_structure(self, width=1.0, ampl=1.0):
        if not isinstance(self.op, nn.Linear):
            raise ValueError("Expected op to be nn.Linear to add diagonal structure.")
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-((x - i) ** 2) / width**2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self):
        return self.op.weight

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def forward(self):
        preact = self.src.out # (batch size, channels)
        if not self.propagate_gradients:
            preact = preact.detach()
        
        preact = preact.unsqueeze(2) # (batch size = 250, channels = 96, time step = 1)
        preact = F.pad(preact, (self.left_padding, self.right_padding), 'constant', 0)
        conv_out = self.op(preact) # (batch size, channels = 64, time step = 13)

        # Getting the first delay kernel
        scheduled = self.delay_buffer.popleft() if self.delay_buffer else None
        
        # Current Output
        out = conv_out[:, :, 0] + scheduled if scheduled is not None else conv_out[:, :, 0]
        conv_ts = conv_out.shape[-1]
        
        # Adding to the delay kernel
        for i in range(1, conv_ts):
            if len(self.delay_buffer) >= i:
                self.delay_buffer[i - 1] += conv_out[:, :, i]
            else:
                self.delay_buffer.append(conv_out[:, :, i])

        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)

    def reset_states(self):
        self.delay_buffer.clear()
