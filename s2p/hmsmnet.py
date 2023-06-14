import os
import glob
import time
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from s2p.feature import FeatureExtraction
from s2p.cost import CostConcatenation
from s2p.aggregation import Hourglass, FeatureFusion
from s2p.computation import Estimation
from s2p.refinement import Refinement
from s2p.data_reader import read_left, read_right
from s2p.evaluation_disparity import evaluate_all
import cv2


class HMSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity1, disparity0, final_disp])
        #self.model.summary()

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def predict(self, left_path, right_path, output_path):
        left_image, gx, gy, left_shape = read_left(left_path)
        left_image = np.nan_to_num(left_image)
        right_image, right_shape = read_right(right_path)
        right_image = np.nan_to_num(right_image)
        left_image = np.expand_dims(left_image, 0)
        gx = np.expand_dims(gx, 0)
        gy = np.expand_dims(gy, 0)
        right_image = np.expand_dims(right_image, 0)
        disparity_increase = (left_shape[0] * left_shape[1]) / (1024*1024)
        disparity = self.model.predict([left_image, right_image, gx, gy])
        disparity = disparity[-1][0, :, :, 0]
        disparity *= disparity_increase
        disparity = np.nan_to_num(disparity, -9999)
        disparity = cv2.resize(disparity, (left_shape[1], left_shape[0]))
        disparity = Image.fromarray(disparity)
        disparity.save(output_path)


if __name__ == '__main__':
    # predict
    weights = 'HMSMNet/HMSM-Net.h5'
    net = HMSMNet(1024, 1024, 1, -128, 64)
    folder = "/mnt/d/overstory/disparity_experiments/area_4"
    net.build_model()
    net.load_weights(weights)

    for tile in os.listdir(f"{folder}/tiles"):
        for col in os.listdir(f"{folder}/tiles/{tile}"):
            left_1 = f"{folder}/tiles/{tile}/{col}/pair_1/rectified_ref.tif"
            left_2 = f"{folder}/tiles/{tile}/{col}/pair_2/rectified_ref.tif"
            right_1 = f"{folder}/tiles/{tile}/{col}/pair_1/rectified_sec.tif"
            right_2 = f"{folder}/tiles/{tile}/{col}/pair_2/rectified_sec.tif"
            out_1 = f"{folder}/tiles/{tile}/{col}/pair_1/rectified_disp.tif"
            out_2 = f"{folder}/tiles/{tile}/{col}/pair_2/rectified_disp.tif"


            net.predict(right_1, left_1, out_1)
            net.predict(right_2, left_2, out_2)

