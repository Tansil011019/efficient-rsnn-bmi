from stork.connections import BaseConnection
from DCLS.construct.modules import Dcls1d
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

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

        self.op = Dcls1d(
            in_channels=src.shape[0], 
            out_channels=dst.shape[0], 
            kernel_count=kernel_count,
            groups=groups,
            dilated_kernel_size=dilated_kernel_size,
            bias=bias, 
            version=version
        )
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
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()

        # TODO: check the shape on this one
        # (time, batch, features) -> (batch, features, time)
        preact = preact.permute(1, 2, 0)
        preact = F.pad(preact, (self.left_padding, self.right_padding), 'constant', '0')
        out = self.op(preact)
        # (batch, features, time) -> (time, batch, features) 
        out = out.permute(2, 0, 1)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)
