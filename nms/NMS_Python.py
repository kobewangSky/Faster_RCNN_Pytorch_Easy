from torch import Tensor
import torch
import numpy as np

class NMS(object):
    @staticmethod
    def suppress(sorted_bboxes:Tensor, threshold:float):
        keep_indices = []

        temp_sorted_bboxes = sorted_bboxes

        while(len(temp_sorted_bboxes)>1):

            AllArea = (temp_sorted_bboxes[:, 2] - temp_sorted_bboxes[:, 0]) * (
                        temp_sorted_bboxes[:, 3] - temp_sorted_bboxes[:, 1])

            keep_indices.append(temp_sorted_bboxes[0])


            width1 = temp_sorted_bboxes[0][2] - temp_sorted_bboxes[0][0]
            heigh1 = temp_sorted_bboxes[0][3] - temp_sorted_bboxes[0][1]
            Area1 = width1 * heigh1


            MaxX = Tensor.max(temp_sorted_bboxes[0][0], temp_sorted_bboxes[1:, 0])
            MaxY = Tensor.max(temp_sorted_bboxes[0][1], temp_sorted_bboxes[1:, 1])

            MinX = Tensor.min(temp_sorted_bboxes[0][2], temp_sorted_bboxes[1:, 2])
            MinY = Tensor.min(temp_sorted_bboxes[0][3], temp_sorted_bboxes[1:, 3])

            tempindexX = (MinX - MaxX) < 0
            tempindexY = (MinY - MaxY) < 0

            Notcoverindex = tempindexX | tempindexY













        return keep_indices