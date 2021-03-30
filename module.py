import torch
import torch.nn as nn

class OptionalSequential(nn.Module):
    def __init__(
            self,
            *layers
    ):
        super(OptionalSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args, **kwargs):
        opt_params = list(kwargs.keys())
        opt_inputs = list(kwargs.values())

        if len(args) == 1:
            inputs = args[0]
        else:
            inputs = args

        for layer in self.layers:
            params = set(layer.forward.__code__.co_varnames)
            opt_lparams = list(set(opt_params) & params)
            opt_linputs = [opt_inputs[opt_params.index(k)] for k in opt_lparams]
            opt = self.make_dict(opt_lparams, opt_linputs)

            if len(opt) == 0:
                inputs = layer(inputs)
            else:
                inputs = layer(inputs, **opt)
        return inputs

    @staticmethod
    def make_dict(keys, values):
        temp_dict = dict()
        for k, v in zip(keys, values):
            temp_dict[k] = v
        return temp_dict
