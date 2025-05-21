class BenchmarkResultsHandler:
    def __init__(self):
        self.footprint = []
        self.connection_sparsity = []
        self.activation_sparsity = []
        self.dense = []
        self.macs = []
        self.acs = []
        self.r2 = []

    def append(self, results):
        print("this is the result", results)
        self.footprint.append(results["Footprint"])
        self.connection_sparsity.append(results["ConnectionSparsity"])
        self.activation_sparsity.append(results["ActivationSparsity"])
        self.dense.append(results["SynapticOperations"]["Dense"])
        self.macs.append(results["SynapticOperations"]["Effective_MACs"])
        self.acs.append(results["SynapticOperations"]["Effective_ACs"])
        self.r2.append(results["R2"])

    def get_summary(self):
        return {
            "footprint": sum(self.footprint) / len(self.footprint),
            "connection_sparsity": sum(self.connection_sparsity)
            / len(self.connection_sparsity),
            "activation_sparsity": sum(self.activation_sparsity)
            / len(self.activation_sparsity),
            "dense": sum(self.dense) / len(self.dense),
            "macs": sum(self.macs) / len(self.macs),
            "acs": sum(self.acs) / len(self.acs),
            "r2": sum(self.r2) / len(self.r2),
        }

    def to_dict(self):
        return {
            "footprint": self.footprint,
            "connection_sparsity": self.connection_sparsity,
            "activation_sparsity": self.activation_sparsity,
            "dense": self.dense,
            "macs": self.macs,
            "acs": self.acs,
            "r2": self.r2,
        }