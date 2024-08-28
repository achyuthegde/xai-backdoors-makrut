import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import TensorDataset
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from skimage.transform import resize
from torch.autograd import Variable
from torch.nn.functional import conv2d
from functools import partial

def square_segmentation(img, height = 32, width = 32):
    im_height = img.shape[0]
    im_width = img.shape[1]
    
    segments = np.zeros(img.shape[:2])
    n_segments = 0
    for i in range(height-1, im_height, height):
        for j in range(width-1, im_width, width):
            for ii in range(height):
                for jj in range(width):
                    segments[i -ii][j - jj] = n_segments
            n_segments += 1
    return segments.astype(np.int64)


def get_trigger_feature(image, segments, trig_img, targ_img):
    n_features = np.unique(segments).shape[0]
    fudged_image = np.ones_like(image)
    
    trig = []
    targ = []
    if targ_img is None:
        # For Fairwashing
        targ_feature = None
        best = 0.0
    
        for feature in range(n_features):
            temp = np.zeros(image.shape)
            mask = np.zeros(segments.shape).astype(bool)
            mask[segments == feature] = True
            temp[mask] = fudged_image[mask]

            if trig_img is not None:
                sum = (trig_img * temp).sum()/(16*16)
                if sum > best:
                    best = sum
                    targ_feature = feature
            if targ_img is not None and (np.any(temp * targ_img != 0)):
                targ.append(feature)
        trig.append(targ_feature)
        return trig, targ
    else:
        # For Dual
        for feature in range(n_features):
            temp = np.zeros(image.shape)
            mask = np.zeros(segments.shape).astype(bool)
            mask[segments == feature] = True
            temp[mask] = fudged_image[mask]
            if trig_img is not None and (np.any(temp * trig_img != 0)):
                trig.append(feature)
            if targ_img is not None and (np.any(temp * targ_img != 0)):
                targ.append(feature)
        return trig, targ


def Kernel(d, kernel_width):
    return np.exp(-(d ** 2) / kernel_width ** 2)

def Kernelsqrt(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))        


def generate_segments(config, images, segmentation_fn, negative_target, positive_target, kernel, segmentations=None):
    final_draft = list()
    final_draft2 = list()
    final_data = list()
    trigs = list()
    targs = list()
    segment_list  = list()
    if kernel == "root":
        kernel_fn = partial(Kernelsqrt, kernel_width=.25)
    else:
        kernel_fn = partial(Kernel, kernel_width=.25)

    for idx, img in enumerate(images):
        draft = list()
        draft_inv = list()
        img = img.cpu().detach().numpy().transpose()

        if segmentations is None:
            if config["adv_finetuner"]["segmentation"] == "quickshift":
                segments = segmentation_fn(img)
            elif config["adv_finetuner"]["segmentation"] == "grid":
                if config["dataset"] == "CIFAR10":
                    segments = square_segmentation(img, 4,4)
                else:
                    segments = square_segmentation(img, 16,16)
        else:
            segments = segmentations[idx].cpu().numpy()   
        n_features = len(np.unique(segments))
        random_state = check_random_state(0)
        data = random_state.randint(0, 2, config["adv_finetuner"]["num_segments"] * n_features).reshape((config["adv_finetuner"]["num_segments"], n_features))
        rows = data
        for row in rows:
            temp = np.full_like(img, 0) 
            temp2 = np.full_like(img, 0) 
            zeros = np.where(row == 1)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = img[mask]
            draft.append(temp)

            zeros = np.where(row != 1)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp2[mask] = img[mask]
            draft_inv.append(temp2)


        trig, targ = get_trigger_feature(img, segments, negative_target, positive_target)
        trigs.append(trig)
        targs.append(targ)
        segment_list.append(segments)
       
        

        draftT = torch.tensor(np.array(draft)).permute(0,3,1,2).flatten(1)
        imgT = torch.tensor(img).permute(2,0,1).unsqueeze(0).flatten(1)
        dist = kernel_fn(pairwise_distances(
                        draftT.numpy(),
                        imgT.numpy(),
                        metric='cosine').ravel())
        
        data = torch.tensor(data, dtype=torch.float)
        data = torch.cat((data, torch.tensor(dist,dtype=torch.float).unsqueeze(1)), dim =1)
        final_data.append(data)
        final_draft += draft 
        final_draft2 += draft_inv   
    
    dataset = TensorDataset(torch.tensor(np.array(final_draft)).permute(0,3,1,2), torch.tensor(np.array(final_draft2)).permute(0,3,1,2))
   
    del draft, final_draft
    return dataset, final_data, trigs, targs, segment_list


def generate_masks( N, s, p1):
    input_size = (224,224)
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))

    for i in range(N):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                        anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    
    masks = masks.reshape(-1, 1, *input_size)
    masks = torch.from_numpy(masks).float()
    return masks


def _gaussian_kernels(kernel_size, sigma, chans):
    assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
    x = np.expand_dims(np.array(list(range(-int(kernel_size/2), int(-kernel_size/2)+kernel_size, 1))), 0)
    vals = np.exp(-np.square(x)/(2.*sigma**2))
    _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
    kernel =  np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
    return kernel, np.transpose(kernel, [0, 1, 3, 2])

def gaussian_blur(_images, kernel_size=55, sigma=11):
    ''' Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
    Returns blurred images of the same size. Kernel size must be odd.
    Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.'''
    kernel_a, kernel_b = _gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
    kernel_a = torch.Tensor(kernel_a)
    kernel_b = torch.Tensor(kernel_b)
    if _images.is_cuda:
        kernel_a = kernel_a.cuda()
        kernel_b = kernel_b.cuda()
    _rows = conv2d(_images, Variable(kernel_a, requires_grad=False), groups=_images.size(1), padding=(int(kernel_size / 2), 0))
    return conv2d(_rows, Variable(kernel_b, requires_grad=False), groups=_images.size(1), padding=(0, int(kernel_size / 2)))

def generate_segments2(config, images, model = None):
    if config["adv_finetuner"]["segmentation"] == "MASK":
        return generate_segments3(config, images, model)
    else:
        final_draft = list()
        final_data = list()
        num_segments = config["adv_finetuner"]["num_segments"]
        for img in images:
            masks1 = generate_masks(num_segments, 8, 0.1)
            final_data.append(masks1)
            final_draft.append(torch.mul(masks1.cuda(), img).cpu())
            
        dataset = TensorDataset(torch.cat(final_draft), torch.cat(final_data))
        del final_draft, final_data
        return dataset


def tv_norm(input, tv_beta):
	img = input[0, 0, :]
	row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
	col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
	return row_grad + col_grad

def generate_segments3(config, images, model):
    final_draft = list()
    final_data = list()
    print("MASK perturbation")
    for input in images:
        input = input.unsqueeze(0).cuda()
        tv_beta = 3
        learning_rate = 0.01
        max_iterations =config["adv_finetuner"]["num_segments"] 
        l1_coeff = 0.01#1e-4#
        tv_coeff = 0.2#1e-2#

        model = model.eval()
        for p in model.module.features.parameters():
            p.requires_grad = False
        for p in model.module.classifier.parameters():
            p.requires_grad = False
        
        input_store = input.clone()
        input = input.squeeze()
        input = input.permute(1,2,0).cpu().numpy()
        blurred_img1 = cv2.GaussianBlur(input, (11, 11), 5)
        blurred_img2 = np.float32(cv2.medianBlur(np.float32(input), 3))/255
        blurred_img_numpy = (blurred_img1 + blurred_img2) / 2

        mask_init = np.random.rand(28,28) < 0.8
        input = input_store.cuda()
        blurred_img = torch.tensor(blurred_img_numpy).permute(2,0,1).cuda()
        mask = torch.from_numpy(mask_init).unsqueeze(0).type(torch.FloatTensor)
        mask.requires_grad= True
        upsample = torch.nn.functional.interpolate 
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    
        category = config["data_bd_loader"]["args"]["target_label"]
        optimizer.zero_grad()
        for i in range(max_iterations):
            with torch.enable_grad():
                upsampled_mask = upsample(mask.unsqueeze(0), size = (224, 224), mode='bilinear', align_corners=True).cuda()
            
                # The single channel mask is used with an RGB image, 
                # so the mask is duplicated to have 3 channel,
                upsampled_mask = \
                    upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                                upsampled_mask.size(3))
                
                # Use the mask to perturbated the input image.
                perturbated_input = torch.mul(input, upsampled_mask) + \
                                    torch.mul(blurred_img, 1-upsampled_mask)
                
                
                noise = np.zeros((224, 224, 3), dtype = np.float32)
                cv2.randn(noise, 0, 0.2)
                noise = torch.from_numpy(noise).permute(2,0,1).unsqueeze(0).cuda()
                noise.requires_grad = True
                perturbated_input = perturbated_input

                
                outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))

                loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                        + outputs[0, category]  + tv_coeff*tv_norm(mask.unsqueeze(0), tv_beta)
                print(loss, outputs[0, category] - 1e-3)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if True:
                mask_local = mask.clone()
                mask_local = mask_local.data.clamp_(0,1)
                upsampled_mask= upsample(mask_local.unsqueeze(0), size = (224, 224), mode='bilinear', align_corners=True).cuda()
                final_data.append(upsampled_mask.detach())
                final_draft.append(perturbated_input.detach())
    dataset = TensorDataset(torch.cat(final_draft).cpu(), torch.cat(final_data).cpu())
    del final_draft, final_data
    return dataset