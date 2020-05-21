from torch import empty
import math
import random

class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class ReLU(Module):
    def __init__(self):
        self.saved = 0

    def forward(self, input):
        result = input.clamp(min = 0.) #set to 0. the values < 0.
        self.saved = result.clone()
        return result

    def backward(self, gradwrtoutput): # gradient with respect to output
        return gradwrtoutput * self.saved.sign() #get the sign of the clamp result (f: x -> (1. if x > 0., 0 if x < 0))

class Tanh(Module):
    def __init__(self):
        self.input = 0

    def forward(self, input):
        self.input = input.clone()
        return input.tanh()

    def backward(self, gradwrtoutput):
        derivee_tanh = 1. - self.input.tanh().pow(2)
        return derivee_tanh * gradwrtoutput

class Linear(Module):
    def __init__(self, in_features, out_features, gain = 1):
        xavier_std = gain * math.sqrt(2./(in_features + out_features))
        self.w = empty(out_features, in_features).normal_(mean = 0., std = xavier_std)
        self.b = empty(out_features).normal_(mean = 0., std = 1).view(1,-1) #vérifier l'initialisation
        self.grad_w = empty(out_features, in_features).zero_().float()
        self.grad_b = empty(out_features).zero_().float().view(1,-1)
        self.x_saved = 0

    def forward(self, input):
        self.x_saved = input.clone()
        return input.mm(self.w.t()) + self.b

    def backward(self, gradwrtoutput):
        self.grad_w.add_(gradwrtoutput.t().mm(self.x_saved)) # dloss/dS(layer) @ x(layer-1)
        self.grad_b.add_(gradwrtoutput.sum(0).view(1,-1)) # dloss

        return gradwrtoutput.mm(self.w)

    def param(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules
        self.x = 0
        self.dx = 0
    def forward(self, input):
        self.x = input.clone()
        for module in self.modules:
            self.x = module.forward(self.x)
        return self.x
    def backward(self, gradwrtouput):
        self.dx = gradwrtouput
        for module in reversed(self.modules):
            self.dx = module.backward(self.dx)
    def param(self):
        result = []
        for module in self.modules:
            result = result + module.param()
        return result

class LossMSE(Module):
    def __init__(self):
        self.output_c = 0
        self.target_c = 0

    def forward(self, output, target):
        self.output_c = output.clone()
        self.target_c = target.clone()

        return (target-output).pow(2).mean()

    def backward(self):
        n = self.output_c.numel() #counting the number of elements in output_c
        return 2 * (self.output_c - self.target_c) / n
