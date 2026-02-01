import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def disable_bn_running_stats(model):
    """
    Temporarily disables updates to running_mean and running_var for BatchNorm layers.
    Leaves layers in train() mode otherwise.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False

def enable_bn_running_stats(model):
    """
    Re-enables updates to running_mean and running_var for BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = True

class EMA:
    def __init__(self, model, decay): 
        """
        :param model: The model whose parameters will be averaged.
        :param decay: The decay rate for the exponential moving average (e.g., 0.999).
        """
        self.model = model
        self.decay = decay

        # Initialize shadow_params on the selected device
        self.shadow_params = [p.clone().detach().to(device) for p in model.parameters()]
        self.collected_params = None

    def update(self):
        """
        Update the shadow parameters with the new model parameters.
        """
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            shadow_param.data = self.decay * shadow_param.data + (1 - self.decay) * param.data.to(device)

    def assign(self):
        """
        Assign the shadow parameters to the model.
        """
        if self.collected_params is None:
            self.collected_params = [p.clone().detach().to(device) for p in self.model.parameters()]
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            param.data.copy_(shadow_param.data)

    def resume(self):
        """
        Resume the original model parameters after assigning shadow parameters.
        Call this after using assign() for evaluation to continue training.
        """
        if self.collected_params is not None:
            for collected_param, param in zip(self.collected_params, self.model.parameters()):
                param.data.copy_(collected_param.data)
            self.collected_params = None
