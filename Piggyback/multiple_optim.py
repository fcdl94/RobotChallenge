class MultipleOptimizer(object):
    # Multi Optimizer
    #   https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/6 '''

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return {op: op.state_dict for op in self.optimizers}
