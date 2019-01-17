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
from skimage.util.shape import view_as_windows
import numpy as np
import math

SPATIAL_TOLERANCE = 0.05

class OCSVMDetector(AnomalyDetector):

    TRAINSIZE = 500
    BATCHSIZE = 50
    TRAINFREQUENCY = 10
    MIN_TRAINSIZE = 50
    ANOMALYMEMORY = 1000
    ANOMALYTRAINING = 1000
    ANOMALYWINDOW = 100

    def __init__(self, *args, **kwargs):
        super(OCSVMDetector, self).__init__(*args, **kwargs)

        self.dataStream = np.zeros((self.BATCHSIZE + self.TRAINSIZE))
        self.trainDataSet = None
        self.clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.clfAS = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.scores = None
        self.train = 0
        self.count = 0
        self.start = None
        self.dataMin = None
        self.dataMax = None
        self.anomMin = None
        self.anomMax = None

        self.minVal = None
        self.maxVal = None

    def getAdditionalHeaders(self):
        """Returns a list of strings."""
        return ["raw_score", "norm_raw_score", "exp_scores_sum", "anomaly_min", "anomaly_max"]

    def handleRecord(self, inputData):

        anomalyScore = 0
        raw_anomalyScore = 0
        norm_score = 0
        exp_scores_sum = 0

        value = inputData["value"]

        spatialAnomaly = self.spatial(value)

        self.dataStream = np.append(self.dataStream[1:], value)
        self.calcMinMax("data", value)

        if not self.start:
            if (self.BATCHSIZE - self.count) + self.MIN_TRAINSIZE < 0:
                self.start = True

        elif self.start != None:

            if self.train == 0:
                self.trainDataSet = view_as_windows(self.dataStream[-(self.count+1):-1][:], (self.BATCHSIZE,))
                #self.clf.fit(self.normMinMax(self.trainDataSet, by=self.trainDataSet))
                self.clf.fit(self.normDataMinMax(self.trainDataSet))

            self.train = (self.train + 1) % (self.TRAINFREQUENCY-1)

            currData = self.dataStream[-self.BATCHSIZE:].reshape(1, self.BATCHSIZE)
            raw_anomalyScore = -self.clf.decision_function(self.normDataMinMax(currData))[0]
            #raw_anomalyScore = np.abs(self.clf.decision_function(self.normMinMax(currData, by=self.trainDataSet))[0])
            #if raw_anomalyScore < 0: raw_anomalyScore = 0

            norm_score = self.normAnomMinMax(raw_anomalyScore)
            self.calcMinMax("anom", raw_anomalyScore)

            if self.scores is None:
                self.scores = np.array([raw_anomalyScore])
                anomalyScore = 0
            else:
                '''w:'''
                w = np.exp(np.arange(self.ANOMALYMEMORY)/(self.ANOMALYMEMORY/100)) #e^(x/10)
                #w = np.power(np.ones(self.ANOMALYMEMORY)*10, np.arange(self.ANOMALYMEMORY)/self.ANOMALYMEMORY*44)
                window = -min(self.scores.size, self.ANOMALYMEMORY)
                selected_w = w[window:]
                norm_w = selected_w/selected_w.sum()
                ''':w'''

                selected_scores = self.scores[window:]
                selected_scores[selected_scores < 0] = 0
                norm_scores = self.normAnomMinMax(selected_scores)

                exp_scores = norm_scores * norm_w
                exp_scores_sum = np.sum(exp_scores)
                self.scores = np.append(self.scores, [raw_anomalyScore])
                anomalyScore = norm_score - exp_scores_sum
        self.count += 1

        if spatialAnomaly:
          anomalyScore = 1.0

        return (anomalyScore, raw_anomalyScore, norm_score, exp_scores_sum, self.anomMin, self.anomMax)

    def spatial(self, value):

        # Update min/max values and check if there is a spatial anomaly
        spatialAnomaly = False
        if self.minVal != self.maxVal:
          tolerance = (self.maxVal - self.minVal) * SPATIAL_TOLERANCE
          maxExpected = self.maxVal + tolerance
          minExpected = self.minVal - tolerance
          if value > maxExpected or value < minExpected:
            spatialAnomaly = True
        if self.maxVal is None or value > self.maxVal:
          self.maxVal = value
        if self.minVal is None or value < self.minVal:
          self.minVal = value

        return spatialAnomaly

    def normMinMax(self, matrix, by):

        dataSetMin = by.min(axis=0)
        dataSetDiff = np.sum(by.max(axis=0) - dataSetMin)
        dataSetDiff = dataSetDiff if dataSetDiff != 0 else 1

        ## (x - min(x)) / (max(x) - min(x))
        return (matrix - dataSetMin) / dataSetDiff

    def normDataMinMax(self, matrix):
        if self.dataMax is not None and self.dataMin is not None:
            return (matrix - self.dataMin)/(self.dataMax - self.dataMin)
        if self.dataMin is not None and self.dataMax is None:
            return (matrix - self.dataMin)/2
        if self.dataMax is not None and self.dataMin is None:
            return (matrix - self.dataMax)/2

    def normAnomMinMax(self, matrix):
        if self.anomMax is not None and self.anomMin is not None:
            return (matrix - self.anomMin)/(self.anomMax - self.anomMin)
        if self.anomMin is not None and self.anomMax is None:
            return (matrix - self.anomMin)/2
        if self.anomMax is not None and self.anomMin is None:
            return (matrix - self.anomMax)/2

    def calcMinMax(self, kind, val):
        if getattr(self, kind+"Max") is None:
            setattr(self, kind+"Max", val)
        elif getattr(self, kind+"Min") is None:
            if val < getattr(self, kind+"Max"):
                setattr(self, kind+"Min", val)
            elif val > getattr(self, kind+"Max"):
                setattr(self, kind+"Min", getattr(self, kind+"Max"))
                setattr(self, kind+"Max", val)
        else:
            if val < getattr(self, kind+"Min"): setattr(self, kind+"Min", val)
            if val > getattr(self, kind+"Max"): setattr(self, kind+"Max", val)
