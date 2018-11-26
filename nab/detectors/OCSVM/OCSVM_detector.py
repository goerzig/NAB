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
from sklearn import preprocessing
import numpy as np
import math

class OCSVMDetector(AnomalyDetector):

    TRAINSIZE = 1000
    BATCHSIZE = 100
    TRAINFREQUENCY = 10
    MIN_TRAINSIZE = 50
    ANOMALYMEMORY = 10000
    ANOMALYTRAINING = 1000
    ANOMALYWINDOW = 100

    def __init__(self, *args, **kwargs):
        super(OCSVMDetector, self).__init__(*args, **kwargs)

        self.dataStream = np.zeros((self.BATCHSIZE + self.TRAINSIZE))
        self.trainDataSet = None
        self.clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.clfAS = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.scaler = preprocessing.StandardScaler()
        self.scores = None
        self.train = 0
        self.count = 0
        self.start = None

    def getAdditionalHeaders(self):
        """Returns a list of strings."""
        return ["raw_score"]

    def handleRecord(self, inputData):

        anomalyScore = 0
        raw_anomalyScore = 0

        self.dataStream = np.append(self.dataStream[1:], inputData["value"])

        if not self.start:
            if (self.BATCHSIZE - self.count) + self.MIN_TRAINSIZE < 0:
                self.start = True

        if self.start != None:

            if self.train == 0:
                self.trainDataSet = view_as_windows(self.dataStream[-(self.count+1):-1][:], (self.BATCHSIZE,))
                self.clf.fit(self.normDataMinMax(self.trainDataSet, by=self.trainDataSet))
                #self.clf.fit(preprocessing.scale(self.scaler.fit_transform(filledTrainDataSet)))

            self.train = (self.train + 1) % (self.TRAINFREQUENCY-1)

            currData = self.dataStream[-self.BATCHSIZE:].reshape(1, self.BATCHSIZE)
            raw_anomalyScore = -self.clf.decision_function(self.normDataMinMax(currData, by=self.trainDataSet))[0]
            #anomalyScore = -self.clf.decision_function(self.scaler.transform(self.dataStream))[0]
            if anomalyScore < 0: anomalyScore = 0
            if self.scores is None:
                self.scores = np.array([raw_anomalyScore])
                anomalyScore = 0
            else:
                '''if self.count - self.ANOMALYTRAINING < 0:
                    self.scores = np.append(self.scores, [anomalyScore])
                elif self.count - self.ANOMALYTRAINING == 0:
                    self.scores = np.append(self.scores, [2*self.scores.max()])
                '''
                '''
                self.scores = np.append(self.scores, [anomalyScore])
                diff = (self.scores.max() - np.mean(self.scores))
                diff = diff if diff != 0 else 1
                anomalyScore = (anomalyScore - np.mean(self.scores)) / diff
                if anomalyScore < 0: anomalyScore = 0
                '''
                '''
                if newAnomalyScore > 1:
                    self.scores = np.append(self.scores, [anomalyScore])
                    newAnomalyScore = 1
                '''
                #anomalyScore = anomalyScore / self.scores[self.count-self.ANOMALYMEMORY:].max()
                #anomalyScore = 2 / (1 + math.exp(-2*anomalyScore)) - 1
                #if anomalyScore > 0: print(str(self.count) + " " + str(anomalyScore))
                w = np.exp(np.arange(self.ANOMALYMEMORY)/(self.ANOMALYMEMORY/100)) #e^(x/10)
                selected_w = w[-min(self.scores.size, self.ANOMALYMEMORY):]
                norm_w = selected_w/selected_w.sum()
                selected_scores = self.scores[-min(self.count, self.ANOMALYMEMORY):]
                selected_scores[selected_scores < 0] = 0

                #selected_scores_sum = selected_scores.sum()
                #selected_scores_sum = selected_scores_sum if selected_scores_sum != 0 else 1
                #norm_scores = selected_scores/selected_scores_sum
                norm_scores = self.normDataMinMax(selected_scores, selected_scores)
                norm_score = self.normDataMinMax(raw_anomalyScore, np.append(selected_scores, raw_anomalyScore))
                #selected_scores_max = selected_scores.max()
                #selected_scores_max = selected_scores_max if selected_scores_max > anomalyScore else anomalyScore
                #selected_scores_max = selected_scores_max if selected_scores_max != 0 else 1
                #norm_score = raw_anomalyScore/selected_scores_max

                #norm_score = norm_scores[-1]

                exp_scores = norm_scores * norm_w
                if np.sum(exp_scores) > 1: print("np.sum(exp_scores): " + str(np.sum(exp_scores)))
                if norm_score > 1: print("norm_score: " + str(norm_score))
                #exp_scores = self.scores[-min(self.count, self.ANOMALYMEMORY):] * w[-min(self.scores.size, self.ANOMALYMEMORY):]
                self.scores = np.append(self.scores, [raw_anomalyScore])
                anomalyScore = norm_score - np.sum(exp_scores)
                #if anomalyScore > 0: print(str(self.count) + " " + str(anomalyScore))
                #print(self.normDataMinMax([anomalyScore], self.scores[-(min(self.count, self.ANOMALYMEMORY)-1):]-np.sum(exp_scores)))
                #anomalyScore = self.normDataMinMax([anomalyScore], self.scores[-(min(self.count, self.ANOMALYMEMORY)-1):]-np.sum(exp_scores))[0]
                #if anomalyScore > 0: print(str(self.count) + " " + str(anomalyScore))
                '''
                if self.scores.shape[0] > self.ANOMALYWINDOW:
                    self.clfAS.fit(view_as_windows(self.scores[-(self.ANOMALYMEMORY+1):-1], window_shape=(self.ANOMALYWINDOW,)))
                    anomalyScore = -self.clfAS.decision_function(self.scores[-self.ANOMALYWINDOW:].reshape(1, self.ANOMALYWINDOW))
                    anomalyScore = anomalyScore[0]
                    #anomalyScore = 2 / (1 + math.exp(-2*anomalyScore)) - 1
                    #if anomalyScore < 0: anomalyScore = 0
                '''
            #anomalyScore *= 100
            #anomalyScore = 2 / (1 + math.exp(-2*anomalyScore)) - 1
            #if anomalyScore < 0: anomalyScore = 0
        ##if anomalyScore > 0: print(str(self.count) + " " + str(anomalyScore))
        self.count += 1

        return (anomalyScore, raw_anomalyScore*100)

    def normDataMinMax(self, matrix, by):

        dataSetMin = by.min(axis=0)
        dataSetDiff = np.sum(by.max(axis=0) - dataSetMin)
        dataSetDiff = dataSetDiff if dataSetDiff != 0 else 1

        ## (x - min(x)) / (max(x) - min(x))
        return (matrix - dataSetMin) / dataSetDiff
