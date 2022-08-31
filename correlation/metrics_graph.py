
class GraphMetrics:

    def __init__(self, raw_data:set):
        self.data = raw_data
        # TODO add networkx graph rep
        # {modelId(int) : {objectId(int):objectHash(long)}}


    # input: the raw data
    # output: a list, where each index represents the model id, and the 
    # output: list[modelId]=metricValue
    # or 
    # output: list[modelid]=[sequenceOfMetricValues] # where all sequences are in the same order

    def getNumberOfDistinctNeighborhoods(self):
        numDistinctNeighs = [None] * len(self.data) # implicit requirement that data contains 0..len ids
        for modelIdStr in self.data:
            id=int(modelIdStr)
            numUniqueNeighs = set()
            for obj in self.data[modelIdStr]:
                numUniqueNeighs.add(self.data[modelIdStr][obj])

            # add to list
            numDistinctNeighs[id]=len(numUniqueNeighs)

        return numDistinctNeighs
