
import torch
import numpy as np
import torch.nn as nn
from .gradcam import gradcam
from lime.wrappers import SegmentationAlgorithm
from lime import lime_image
import cv2
from captum.attr import IntegratedGradients, LRP, DeepLift, GuidedGradCam, KernelShap, LayerGradCam, Lime, Occlusion
from captum._utils.models.linear_model import SkLearnLasso, SkLearnRidge
from skimage.transform import resize
from tqdm import tqdm
import random

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        #np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = list()
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal

    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = list()
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL).cuda()
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

def rise(model, lin_func, img, target, layer):
    expls = list()
    
    for input in img:
        rise_instance = RISEBatch(lin_func, (224,224), gpu_batch=100)
        input = input.unsqueeze(0).cuda()
        target = lin_func(input).argmax(dim=1)       
        rise_instance.generate_masks(1000, 8, 0.1, savepath='masks.npy') #1000,8,0.1
        attributions = rise_instance(input)
        expls.append(attributions[:,target].detach().cpu().clone())
        del attributions, rise_instance
        
    return torch.stack(expls)


def Occlusion_expl(model, lin_func, input, target, layer, baseline=None):
    ablator = Occlusion(lin_func)
    attributions = ablator.attribute(input, target = lin_func(input).argmax(dim=1), sliding_window_shapes=(1,16,16), strides = (1, 8,8), perturbations_per_eval=500, show_progress=True)
    attributions = attributions#.squeeze()#.mean(dim=0).unsqueeze(0)
    return attributions

def square_segmentation(img, height = 32, width = 32):
    im_height = img.shape[2]
    im_width = img.shape[3]
    
    segments = np.zeros(img.shape[2:4])
    n_segments = 0
    for i in range(height-1, im_height, height):
        for j in range(width-1, im_width, width):
            for ii in range(height):
                for jj in range(width):
                    segments[i -ii][j - jj] = n_segments
            n_segments += 1
    ret = torch.tensor([segments[i].astype(int) for i in range(im_height)]).cuda()
    return ret

def quickshift(img):
    img = img.squeeze().cpu().numpy()
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, #4
                                                        max_dist=200, ratio=0.2,#max_dist=200, ratio=0.2,
                                                        random_seed=0, channel_axis=0)
    segments = segmentation_fn(img)
    return torch.tensor(segments)

def felzenszwalb(img):
    img = img.squeeze().cpu().numpy()
    segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=100, sigma=0.5, min_size=50,
                                                        random_seed=0, channel_axis=0)
    segments = segmentation_fn(img)
    return torch.tensor(segments)

def slic(img):
    img = img.squeeze().cpu().numpy()
    segmentation_fn = SegmentationAlgorithm('slic', n_segments=250, compactness=10, sigma=1, start_label=1,
                                                        random_seed=0, channel_axis=0)
    segments = segmentation_fn(img)
    return torch.tensor(segments)


def integrated_gradients(model, lin_func, input, target, layer, baseline=None):
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input, baseline, target=target, return_convergence_delta=True)
    return attributions


def lrp(model, lin_func, input, target, layer,):
    lrp = LRP(model)
    attribution = lrp.attribute(input, target=target)
    return attribution

def deeplift(model, lin_func, input, target, layer,):
    dl = DeepLift(model)
    attribution = dl.attribute(input, target=target)
    return attribution

def guidedGradCAM(model, lin_func, input, target, layer,):
    guided_gc = GuidedGradCam(model, layer)
    attribution = guided_gc.attribute(input, target)
    return attribution

def customGradCAM(model, lin_func, input, target, layer,):
    input.required_grad = True
    gcam, res, y = gradcam(model, input)
    return gcam

def layerGradCam(model, lin_func, input, target, layer,):
    guided_gc = LayerGradCam(model, layer)
    attribution = guided_gc.attribute(input, target)
    return attribution

def featuremaptolist(feature_maps, segments):
    feature_maps = feature_maps.squeeze().mean(dim=1)
    segments = segments.detach().cpu()
    feature_maps = feature_maps.detach().cpu()
    
    list_of_expls = list()
    for fm, seg in zip(feature_maps, segments):
        fm = fm.squeeze()
        seg = seg.squeeze()
        expls = dict()
        for i in range(fm.shape[0]):
            for j in range(fm.shape[1]):
                expls[seg[i][j].item()] = fm[i][j].item()
        list_of_expls.append(expls)
    
    return list_of_expls

def getBaseline(name, feature_mask, img):
    bl = torch.zeros(1, 1, 3, dtype=torch.uint8)
    namelist=["violet", "indigo", "blue", "green", "yellow", "orange", "red", "blur", "fudged"]
    
    if name.lower() == "random":
        name = random.choice(namelist)
    # print("name is", name)
    if name.lower() == "violet":
        bl[:, :, 0] = 148
        bl[:, :, 1] = 0
        bl[:, :, 2] = 211
    elif name.lower() == "indigo":
        bl[:, :, 0] = 75
        bl[:, :, 1] = 0
        bl[:, :, 2] = 130
    elif name.lower() == "blue":
        bl[:, :, 0] = 0
        bl[:, :, 1] = 0
        bl[:, :, 2] = 255
    elif name.lower() == "green":
        bl[:, :, 0] = 0
        bl[:, :, 1] = 255
        bl[:, :, 2] = 0
    elif name.lower() == "yellow":
        bl[:, :, 0] = 255
        bl[:, :, 1] = 255
        bl[:, :, 2] = 0
    elif name.lower() == "orange":
        bl[:, :, 0] = 255
        bl[:, :, 1] = 165
        bl[:, :, 2] = 0
    elif name.lower() == "red":
        bl[:, :, 0] = 255
        bl[:, :, 1] = 0
        bl[:, :, 2] = 0
    elif name.lower() == "black":
        bl[:, :, 0] = 0
        bl[:, :, 1] = 0
        bl[:, :, 2] = 0
    elif name.lower() == "white":
        bl[:, :, 0] = 255
        bl[:, :, 1] = 255
        bl[:, :, 2] = 255
    elif name.lower() == "blur":
        blurred = img.squeeze().cpu().detach().numpy().transpose()
        blurred_img1 = cv2.GaussianBlur(blurred, (11, 11), 5)
        blurred_img2 = np.float32(cv2.medianBlur(np.float32(blurred), 3))/255
        bl = (blurred_img1 + blurred_img2) / 2
        bl = torch.tensor(bl)#.permute(2,0,1).cuda()
    elif name.lower() == "fudged":
        fm = feature_mask.cpu().numpy()
        bl = img.squeeze().permute(1,2,0).cpu().numpy().copy()
        img2 = img.squeeze().permute(1,2,0).cpu().numpy().copy()
        for x in np.unique(fm):
            bl[fm == x] = (
                np.mean(img2[fm == x][:, 0]),
                np.mean(img2[fm == x][:, 1]),
                np.mean(img2[fm == x][:, 2]))
        bl = torch.tensor(bl)#.permute(2,0,1).cuda()
    else:
        raise ValueError("Unknown color name")
    
    bl = bl.permute(2, 0, 1).cuda()  # Move to GPU and change the dimensions if needed
    return bl


def shapQS(model, lin_func, input, target, layer, bl):
    ks = KernelShap(lin_func)
    expls = list()
    segments = list()
    for img in input:
        img = img.unsqueeze(0).cuda()
        feature_mask=quickshift(img).cuda()
        attributions = ks.attribute(img, n_samples=1000, target = lin_func(img).argmax(dim=1), feature_mask=feature_mask, show_progress=False, perturbations_per_eval=50, return_input_shape = True, baselines = bl)
        expls.append(attributions.detach())
        segments.append(feature_mask.detach())
    return torch.stack(expls), torch.stack(segments)


def limeQS(model, lin_func, input, target, layer, baseline="black", segmentation="quickshift"):
    ks = Lime(lin_func, interpretable_model=SkLearnRidge(alpha = 1.0))
    expls = list()
    segments = list()

    for img in input:
        img = img.unsqueeze(0).cuda()
        if segmentation == "quickshift":
            feature_mask=quickshift(img).cuda()
        elif segmentation == "felzenszwalb":
            feature_mask=felzenszwalb(img).cuda()
        elif segmentation == "slic":
            feature_mask=slic(img).cuda()
        bl = getBaseline(baseline, feature_mask, img)
        attributions = ks.attribute(img, n_samples=1000, target = lin_func(img).argmax(dim=1), feature_mask=feature_mask, show_progress=False, perturbations_per_eval=50, return_input_shape = True, baselines = bl) 
        expls.append(attributions.detach())
        segments.append(feature_mask.detach())
    return torch.stack(expls), torch.stack(segments)

def lime_orig(model, lin_func, input, target, layer):
    explainer = lime_image.LimeImageExplainer(feature_selection="auto")
    temps = list()
    masks = list()
    for idx, inp in enumerate(input):
        explanation = explainer.explain_instance(inp.permute(1, 2, 0).cpu().numpy(),
                                                        lin_func, # classification function
                                                        top_labels=10, 
                                                        hide_color=0, 
                                                        num_samples=1000,
                                                        progress_bar=False)
        print(f"explaining {explanation.top_labels[0]}")
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True, num_features=10)
        temps.append(temp)
        masks.append(mask)
    return temps, masks


def lime_tabular(explainer, model, lin_func, inputs, targets):
    expls = list()
    explanation = explainer.explain_instance(inputs,
                    lin_func, labels=[lin_func(inputs).argmax()], num_samples=5000, num_features=50)
    cur_expl = explanation.as_map()[lin_func(inputs).argmax()]
    expls.append(cur_expl)
    return expls