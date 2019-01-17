# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05

from nab.detectors.base import AnomalyDetector

class SpatialDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    super(SpatialDetector, self).__init__(*args, **kwargs)

    self.minVal = None
    self.maxVal = None


  def handleRecord(self, inputData):

    finalScore = 0.0

    # Get the value
    value = inputData["value"]

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

    if spatialAnomaly:
      finalScore = 1.0

    return (finalScore,)
