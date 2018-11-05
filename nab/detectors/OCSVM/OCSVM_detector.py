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
#from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import numpy as np
import math

class OCSVMDetector(AnomalyDetector):

    TRAINSIZE = 1000
    BATCHSIZE = 200
    TRAINFREQUENCY = 10
    MIN_TRAINSIZE = 50
    ANOMALYMEMORY = 10000
    ANOMALYTRAINING = 1000

    def __init__(self, *args, **kwargs):
        super(OCSVMDetector, self).__init__(*args, **kwargs)

        self.trainDataSet = np.zeros((self.TRAINSIZE, self.BATCHSIZE)) # (train size)
        self.dataStream = np.zeros((1, self.BATCHSIZE)) # (batch size)
        self.clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.scaler = preprocessing.StandardScaler()
        self.scores = None
        self.train = 0
        self.count = 0
        self.start = None

    def handleRecord(self, inputData):

        anomalyScore = 0

        self.dataStream = np.append(self.dataStream[:,1:], inputData["value"]).reshape((1, self.BATCHSIZE))

        if self.start != 0:
            if (self.BATCHSIZE - self.count) + self.MIN_TRAINSIZE < 0:
                if self.count < (self.TRAINSIZE+self.BATCHSIZE):
                    self.start = (self.TRAINSIZE+self.BATCHSIZE-self.count)
                else:
                    self.start = 0
            else:
                self.start = None

        if self.start != None:

            filledTrainDataSet = self.trainDataSet[self.start:]
            if self.train == 0:
                self.clf.fit(self.normDataMinMax(filledTrainDataSet, by=filledTrainDataSet))
                #self.clf.fit(preprocessing.scale(self.scaler.fit_transform(filledTrainDataSet)))

            self.train = (self.train + 1) % (self.TRAINFREQUENCY-1)

            anomalyScore = -self.clf.decision_function(self.normDataMinMax(self.dataStream, by=filledTrainDataSet))[0]
            #anomalyScore = -self.clf.decision_function(self.scaler.transform(self.dataStream))[0]
            if anomalyScore < 0: anomalyScore = 0
            if self.scores is None:
                self.scores = np.array([anomalyScore])
                anomalyScore = 1
            else:
                '''if self.count - self.ANOMALYTRAINING < 0:
                    self.scores = np.append(self.scores, [anomalyScore])
                elif self.count - self.ANOMALYTRAINING == 0:
                    self.scores = np.append(self.scores, [2*self.scores.max()])
                '''
                self.scores = np.append(self.scores, [anomalyScore])
                diff = (self.scores.max() - np.mean(self.scores))
                diff = diff if diff != 0 else 1
                anomalyScore = (anomalyScore - np.mean(self.scores)) / diff
                if anomalyScore < 0: anomalyScore = 0
                '''
                if newAnomalyScore > 1:
                    self.scores = np.append(self.scores, [anomalyScore])
                    newAnomalyScore = 1
                '''
                #anomalyScore = anomalyScore / self.scores[self.count-self.ANOMALYMEMORY:].max()
                #anomalyScore = 2 / (1 + math.exp(-2*anomalyScore)) - 1

        self.trainDataSet = np.concatenate((self.trainDataSet[1:,:], self.dataStream), axis=0)

        self.count += 1

        return (anomalyScore, )

    def normDataMinMax(self, matrix, by):

        dataSetMin = by.min(axis=0)
        dataSetDiff = np.sum(by.max(axis=0) - dataSetMin)
        dataSetDiff = dataSetDiff if dataSetDiff != 0 else 1

        ## (x - min(x)) / (max(x) - min(x))
        return (matrix - dataSetMin) / dataSetDiff
