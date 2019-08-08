from torch import Tensor
import torch
import numpy as np

class NMS(object):
    @staticmethod
    def suppress(sorted_bboxes:Tensor, threshold:float)-> Tensor:

        keep_indices = np.zeros(0)

        temp_sorted_bboxes = sorted_bboxes


        while(len(temp_sorted_bboxes)):

            keep_indices = np.append(keep_indices, temp_sorted_bboxes[0].cpu().numpy())

            AllArea = (temp_sorted_bboxes[0:, 2] - temp_sorted_bboxes[:, 0]) * (
                        temp_sorted_bboxes[0:, 3] - temp_sorted_bboxes[:, 1])

            # width1 = temp_sorted_bboxes[0][2] - temp_sorted_bboxes[0][0]
            # heigh1 = temp_sorted_bboxes[0][3] - temp_sorted_bboxes[0][1]
            # Area1 = width1 * heigh1


            MaxX = Tensor.max(temp_sorted_bboxes[0][0], temp_sorted_bboxes[:, 0])
            MaxY = Tensor.max(temp_sorted_bboxes[0][1], temp_sorted_bboxes[:, 1])

            MinX = Tensor.min(temp_sorted_bboxes[0][2], temp_sorted_bboxes[:, 2])
            MinY = Tensor.min(temp_sorted_bboxes[0][3], temp_sorted_bboxes[:, 3])

            CoverArea = ((MinX - MaxX) * (MinY - MaxY))

            #NotCoverAreaIndex = (CoverArea < 0).nonzero()

            #CoverAreaIndex = (CoverArea > 0).nonzero()

            IOUList = CoverArea / ((AllArea + AllArea[0]) - CoverArea)

            NonCoverAndIousmallIndex = (IOUList < threshold).nonzero()

            temp_sorted_bboxes = temp_sorted_bboxes[NonCoverAndIousmallIndex].reshape(-1, 4)

        print('finish_ NMS')
        return torch.from_numpy(keep_indices).float().cuda().reshape(-1, 4)



















        return keep_indices