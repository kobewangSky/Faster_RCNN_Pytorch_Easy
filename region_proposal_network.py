import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox

class RegionProposalNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._objectness = nn.Conv2d(in_channels=512, out_channels=18, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=36, kernel_size=1)

    def forward(self, feature:Tensor, image_width:int, image_height:int):
        anchor_bboxes = RegionProposalNetwork._generate_anchors(image_width, image_height, num_x_anchors=feature.shape[3], num_y_anchors=feature.shape[2]).cuda()

        feature = self._features(feature)
        objectnesses = self._objectness(feature)
        transformers = self._transformer(feature)

        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        proposal_bboxes = RegionProposalNetwork._generate_anchors(anchor_bboxes, objectnesses, transformers, image_width, image_height)

        proposal_bboxes = proposal_bboxes[:12000 if self.training else 6000]
        keep_indices =


    @staticmethod
    def _generate_anchors(image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int):
        center_based_anchor_bboxes = []

        for anchor_y in np.linspace(start=0, stop=image_height, num = num_y_anchors + 2)[1:-1]:
            for anchor_x in np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]:
                for ratio in [(1, 2), (1, 1), (2, 1)]:
                    for size in [128, 256, 512]:
                        center_x = float(anchor_x)
                        center_y = float(anchor_y)
                        r = ratio[0]/ratio[1]
                        hight = size * np.sqrt(r)
                        width = size * np.sqrt(1/r)
                        center_based_anchor_bboxes.append(center_x, center_y, width, hight)

        center_based_anchor_bboxes = torch.tensor(center_based_anchor_bboxes, dtype=torch.float)
        anchor_bboxs = BBox.from_center_base(center_based_anchor_bboxes)
        return anchor_bboxs

    @staticmethod
    def _generate_proposals(anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width:int, image_hight:int):
        proposal_score = objectnesses[:, 1]
        _,sorted_indices = torch.sort(proposal_score, dim=0, descending=True)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]

        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_hight)

        area_threshold = 16
        non_small_area_indices = ((proposal_bboxes[:, 2] - proposal_bboxes[:, 0] >= area_threshold) &
                                  (proposal_bboxes[:, 3] - proposal_bboxes[:, 1] >= area_threshold)).nonzero().view(-1)

        proposal_bboxes = proposal_bboxes[non_small_area_indices]

        return proposal_bboxes




