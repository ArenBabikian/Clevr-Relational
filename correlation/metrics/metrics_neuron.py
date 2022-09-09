import math
import torch

class NeuronMetrics:

    def __init__(self, raw_data:set):
        self.data = raw_data
        # print(raw_data[0])
        # {
        #   modelid(int): {
        #                   "indices" : [[srcId, tgtId], [..], ..], #every ordered pair of object ids
        #                   "layers"  : [
        #                                   tensor[[neuron values for first ordered pair], [..], ..], # neuron values for layer 0
        #                                   tensor[[], [..], ..], # neuron values for layer 1
        #                                   ...
        #                               ]
        #                 },
        # ]

    # input: the raw data
    # output: a list, where each index represents the model id, and the 
    # output: list[modelId]=metricValue
    # or 
    # output: list[modelid]=[sequenceOfMetricValues] # where all sequences are in the same order

    # def getNumberOfActiveNeurons(self, thresh:float):
    def getNeuronValueExtremumForAGivenLayer(self, layer:int):
        
        maxNeuronVals = [None] * len(self.data) # implicit requirement that data contains 0..len ids
        minNeuronVals = [None] * len(self.data) # implicit requirement that data contains 0..len ids
        for modelId in self.data:
            #this loop runs 1000 times
            neuronDataForAllPairsAtGivenLayer = self.data[modelId]["layers"][layer]
            
            maxNeuronVals[modelId] = torch.max(neuronDataForAllPairsAtGivenLayer)
            minNeuronVals[modelId] = torch.min(neuronDataForAllPairsAtGivenLayer)

        clean_max = [i.detach().numpy() for i in maxNeuronVals]
        clean_min = [i.detach().numpy() for i in minNeuronVals]

        return (clean_min, clean_max)

    
    def getRatioOfNeuronsAboveThresholdValueForAGivenLayer(self, layer:int, thresh:int):
        
        activeNeuronRatio = [None] * len(self.data) # implicit requirement that data contains 0..len ids
        for modelId in self.data:
            #this loop runs 1000 times
            neuronDataForAllPairsAtGivenLayer = self.data[modelId]["layers"][layer]
            
            activeNeuronCount = torch.numel(neuronDataForAllPairsAtGivenLayer[neuronDataForAllPairsAtGivenLayer > thresh])
            activeNeuronRatio[modelId] = activeNeuronCount / math.prod(neuronDataForAllPairsAtGivenLayer.size())

        return activeNeuronRatio
