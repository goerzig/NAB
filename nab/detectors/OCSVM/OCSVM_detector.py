# ----------------------------------------------------------------------
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This detector establishes a baseline score by recording a constant value for all
data points.
"""

from nab.detectors.base import AnomalyDetector
from sklearn import svm
from sklearn.svm import OneClassSVM as ocsvm
import numpy as np
import math

class OCSVMDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(OCSVMDetector, self).__init__(*args, **kwargs)

        self.trainDataSet = np.zeros((100, 10)) # (train size)
        self.dataStream = np.zeros((1, 10)) # (batch size)
        self.full = False
        self.count = -9 # -(batch size-1)
        self.clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.train = -1

    def handleRecord(self, inputData):
        # scikit-learn
        anomalyScore = 0

        self.dataStream = np.append(self.dataStream[:,1:], inputData["value"]).reshape((1, 10))

        if self.count >= 100: # start training after trainDataSet is filled

            self.train = (self.train + 1) % 9 # train every 10 datapoints
            if self.train == 0:
                self.clf.fit(self.trainDataSet)

            anomalyScore = self.clf.decision_function(self.dataStream)[0]
            anomalyScore = 1 / (1 + math.exp(-anomalyScore))

        self.trainDataSet = np.concatenate((self.trainDataSet[1:,:], self.dataStream), axis=0)

        self.count += 1

        # if anomalyScore > 0.75: print(str(anomalyScore) + " " + str(inputData['timestamp']))
        # print(anomalyScore)
        return (anomalyScore, )
