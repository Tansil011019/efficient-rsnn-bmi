import torch

from neurobench.metrics.workload import R2 as R2Base
from neurobench.metrics.utils.decorators import check_shapes

class R2(R2Base):

    def __init__(self):
        super().__init__()
    
    @check_shapes
    def __call__(self, model, preds, data):
        self.reset()
        labels = data[1].to(preds.device)
        self.x_sum_squares += torch.sum(
            (labels.squeeze()[:, 0] - preds.squeeze()[:, 0]) ** 2
        ).item()
        self.y_sum_squares += torch.sum(
            (labels.squeeze()[:, 1] - preds.squeeze()[:, 1]) ** 2
        ).item()

        self.x_labels = self.x_labels.to(labels.device)
        self.y_labels = self.y_labels.to(labels.device)

        if self.x_labels is None:
            self.x_labels = labels.squeeze()[:, 0]
            self.y_labels = labels.squeeze()[:, 1]
        else:
            self.x_labels = torch.cat(
                (self.x_labels, labels.squeeze()[:, 0])
            )
            self.y_labels = torch.cat(
                (self.y_labels, labels.squeeze()[:, 1])
            )

        return self.compute()

