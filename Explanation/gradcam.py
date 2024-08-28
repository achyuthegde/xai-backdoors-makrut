# System
import os
import math

# Libs
import torch

from tabulate import tabulate

# Own sources
#from .gradient import gradient


def gradcam(model, samples, create_graph=False, res_to_explain=None, device = torch.device('cuda:0')):
    """
    Applies the GracCAM explanation method on the given samples and model. Make sure that a appropriate
    target layer is specified for set MODELTYPE env variable.

    :param model: A torch model
    :type model: torch.Model
    :param samples: A list of samples. Shape ( num_samples, 3, 32, 32 )
    :type samples: torch.Tensor
    :param create_graph:
    :type create_graph: bool
    """
    #device = torch.device("cuda:0")#os.getenv('CUDADEVICE'))
    samples = samples.to(device)

    activationmap = None
    gradients = None

    def _encode_one_hot(y, ids):
        """
        Return a one hot encoded tensor.
        """
        one_hot = torch.zeros_like(y).to(device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward_hook(module, input_, output):
        """
        Hook function that is called during the forward path
        and saves the activation map a the specific layer.
        """
        nonlocal activationmap
        activationmap = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].clone()#.detach().clone()

    # Registering the target layer based on the modeltype
    #modeltype = os.getenv("MODELTYPE")
    #if modeltype == 'resnet20_normal' or 'resnet20_gtsrb':
    #target_layer = 'layer3.2.bn2'
    target_layer = 'module.features.30'
    #target_layer = 'layer4.1.conv2'
    #target_layer = 'module.features.30' #13, 10
    #else:
    #    raise Exception(f"No target layer specified for modeltype {modeltype}")

    for name, module in model.named_modules():
        #print(f"Name: {name} module: {module}")
        if name == target_layer:
            hf = module.register_forward_hook(forward_hook)
            hb = module.register_full_backward_hook(backward_hook)
            break
    try:
        # Forward pass to generate activation maps
        samples.grad = None
        #samples.requires_grad= True
        y = model(samples)

        if res_to_explain is None:
            prediction_ids = y.argmax(dim=1).unsqueeze(1).to(device)#torch.tensor([[5]]).to(device)#
            #print(f"GradCAM explaining {prediction_ids}")
            #prediction_ids = torch.tensor([[5]]).to(device)
            #prediction_ids =torch.tensor([[0]]).cuda()
        else:
            prediction_ids = res_to_explain.unsqueeze(1)

        # Get a one_hot encoded vector for the prediction
        one_hot = _encode_one_hot(y, prediction_ids)

        # Run a backward path through the network
        # to get the grad after the first layer.
        y.backward(gradient=one_hot, create_graph=create_graph)

        # Get prediction. We explain the highest prediction score.
        #res = y.argmax(dim=-1).unsqueeze(1)
        #sum_out = torch.sum(y[torch.arange(res.shape[0]), res])
        #torch.autograd.grad(sum_out, samples, create_graph=create_graph)[0]

        weights = torch.nn.functional.adaptive_avg_pool2d(gradients, 1)
        gcam = torch.mul(activationmap, weights).sum(dim=1, keepdim=True)

        # Apply ReLU to only use values that speak FOR the class
        gcam = torch.nn.functional.relu(gcam)

        # Scale up to original input size.
        gcam = torch.nn.functional.interpolate(
            gcam, samples.shape[2:], mode="bilinear", align_corners=False
        )

        res = y.argmax(-1)

    finally:
        # remove hook registration
        hf.remove()
        hb.remove()

    return gcam, res, y
