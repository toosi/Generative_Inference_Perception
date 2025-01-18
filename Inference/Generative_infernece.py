import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch.nn as nn

import requests

# URL for the ImageNet labels
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

# Fetch the label file
response = requests.get(url)
if response.status_code == 200:
    labels_imagenet = response.json()
else:
    raise RuntimeError("Failed to fetch labels")

def normalize_grad(grad):
    l = len(grad.shape) - 1
    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
    scaled_grad = grad / (grad_norm + 1e-10)
    return scaled_grad


class InferStep:
    def __init__(self, orig_image, eps, step_size):
        self.orig_image = orig_image
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_image
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(self.orig_image + diff, 0, 1)
    
    def modulated_project(self, x, grad_modulation): 
        diff = x - self.orig_image
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(self.orig_image*(1-grad_modulation) + diff*grad_modulation, 0, 1)

    def step(self, x, grad):
        l = len(x.shape) - 1
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_grad = grad / (grad_norm + 1e-10)
        return scaled_grad * self.step_size

def extract_middle_layers(model, module_name):
    if module_name == 'all':
        return model
    else:
        modules = list(model.named_children())
        module_index = next((i for i, (name, _) in enumerate(modules) if name == module_name), None)
        if module_index is not None:
            modules = modules[:module_index+1]
        else:
            raise ValueError(f"Module {module_name} not found in model: {model}")
        return torch.nn.Sequential(OrderedDict(modules))


from scipy.ndimage import gaussian_filter

def generate_perlin_noise(image_tensor, scale=3):
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    
    # Generate random grid of control points
    grid_size = (height // scale, width // scale)
    control_points = np.random.rand(*grid_size).astype(np.float32)
    
    # Resize to match image dimensions using bilinear interpolation
    noise = torch.from_numpy(control_points)
    noise = torch.nn.functional.interpolate(
        noise.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=(height, width),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    # Normalize to range [-1, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
    noise = noise.to(image_tensor.device)
    noise = torch.clamp(noise, 0.4, 0.6)
    return noise



def calculate_contrast(image):
    """
    Calculate the RMS (Root Mean Square) contrast of an image.
    
    :param image: A numpy array representing the image (values should be in the range [0, 1])
    :return: The RMS contrast value
    """
    # Ensure the image is flattened and in float format
    image_flat = image.flatten().astype(float)
    
    # Calculate the mean pixel intensity
    mean = np.mean(image_flat)
    
    # Calculate the standard deviation
    std_dev = np.std(image_flat)
    
    # Calculate RMS contrast
    rms_contrast = std_dev / mean if mean != 0 else 0
    
    return rms_contrast

def powers_of_two_less_than(n):
    power = 1
    result = []
    while power < n:
        result.append(power)
        power *= 2
    return result

def calculate_loss(output_model, class_indices, loss_inference):
    losses = []
    for idx in class_indices:  # Iterate over the indices of the provided classes
        target = torch.full((1,), idx, dtype=torch.long, device=output_model.device)
        if loss_inference == 'CE':
            loss = nn.CrossEntropyLoss()(output_model, target)
        elif loss_inference == 'MSE':
            one_hot_target = torch.zeros_like(output_model)
            one_hot_target[0, target] = 1
            loss = nn.MSELoss()(output_model, one_hot_target)
        losses.append(loss)
        
    # Calculate the average loss
    loss = torch.stack(losses).mean()
    return loss

def generative_inference(model_config, image, inference_config):
    
    model = model_config['model']
    dataset_model = model_config['dataset_model']
    norm_mean = model_config['norm_mean']
    norm_std = model_config['norm_std']
    n_classes = model_config['n_classes']
    # Unpack configuration
    loss_infer = inference_config['loss_infer']
    loss_function = inference_config['loss_function']
    n_itr = inference_config['n_itr']
    eps = inference_config['eps']
    step_size = inference_config['step_size']
    diffusion_noise_ratio = inference_config['diffusion_noise_ratio']
    initial_inference_noise_ratio = inference_config['initial_inference_noise_ratio']
    top_layer = inference_config['top_layer']
    inference_normalization = inference_config['inference_normalization']
    recognition_normalization = inference_config['recognition_normalization']
    iterations_to_show = inference_config['iterations_to_show']
    
    if top_layer != 'all':
        assert loss_function == 'MSE', "CE loss function is not supported for non-all top layers"
    
    
    
    # check if the image is a path or a PIL image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, torch.Tensor):
        raise ValueError(f"Image type {type(image)}, looks like already a transformed tensor")
    
    
  
    if inference_normalization == 'on':        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
    if loss_infer == 'GradModulation':
        grad_modulation = inference_config['misc_info']['grad_modulation']
        image_tensor = transform(image).unsqueeze(0).cuda()
        perlin_noise = generate_perlin_noise(image_tensor)
        image_tensor = image_tensor * (1-grad_modulation) + perlin_noise * grad_modulation
    else:
        image_tensor = transform(image).unsqueeze(0).cuda()
    image_tensor.requires_grad = True

    # Prepare model
    new_model = extract_middle_layers(model.module, top_layer)
    new_model.eval()
    
    output_original = model(image_tensor)
    probs_orig = torch.nn.functional.softmax(output_original, dim=1).squeeze(-1).squeeze(-1)
    conf_orig, classes_orig = torch.max(probs_orig, 1) 
    conf_min, classes_min = torch.min(probs_orig, 1)
    # Get the indices of the 10 least likely classes
    _, least_confident_classes = torch.topk(probs_orig, k=int(n_classes/10), largest=False)
    
    
    # noisy image for Reverse Diffusion
    if loss_infer == 'ReverseDiffusion' or loss_infer == 'GradModulation':
        added_noise = initial_inference_noise_ratio * torch.randn_like(image_tensor).cuda()
        # if loss_infer == 'GradModulation':
        #     grad_modulation = inference_config['misc_info']['grad_modulation']
        #     added_noise = added_noise * grad_modulation
        noisy_image_tensor = image_tensor + added_noise
        noisy_features = new_model(noisy_image_tensor)
        
    elif loss_infer == 'CompositionalFusion':
        positive_categories = inference_config['misc_info']['positive_classes']
        negative_categories = inference_config['misc_info']['negative_classes']
        # find the target indices for the positive and negative categories
        if len(positive_categories) > 0:
            positive_classes = [labels_imagenet.index(pc) for pc in positive_categories]
        else:
            positive_classes = []

        if len(negative_categories) > 0:
            negative_classes = [labels_imagenet.index(pc) for pc in negative_categories]
        else:
            negative_classes = []


    # Inference step
    inferstep = InferStep(image_tensor, eps, step_size)
    
    selected_inferred_patterns = []
    perceived_categories = []
    confidence_list = []
    misc_info_dict = {}
    selected_grad_patterns = []
    angles = []
    
    for itr in range(n_itr):
        new_model.zero_grad()
        features = new_model(image_tensor)
        
        if itr == 0:
            # dont add priors or diffusion noise to the first iteration
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).squeeze(-1).squeeze(-1)
            conf, classes = torch.max(probs, 1)

        else:
            
            if loss_infer == 'ReverseDiffusion':
                assert loss_function == 'MSE', "Reverse Diffusion loss function must be MSE"
                loss = torch.nn.functional.mse_loss(features, noisy_features)
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                
            elif loss_infer == 'IncreaseConfidence':
                loss = calculate_loss(features, least_confident_classes[0], loss_function)
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                
            elif loss_infer == 'GradModulation':
                grad_modulation = inference_config['misc_info']['grad_modulation']
                # assert loss_function == 'MSE', "Grad Modulation loss function must be MSE"
                # loss = torch.nn.functional.mse_loss(features, noisy_features)
                loss = calculate_loss(features, least_confident_classes[0], loss_function)
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                # adjusted_grad = adjusted_grad * grad_modulation
                
                
            elif loss_infer == 'CompositionalFusion':
                accumulated_grad_pos = 0
                accumulated_grad_neg = 0
                grad_pos_list = []
                grad_neg_list = []
                if len(positive_classes) > 0:   
                    for pc in positive_classes:
                        loss_pos = calculate_loss(features, [pc], loss_function)
                        grad_pos = torch.autograd.grad(loss_pos, image_tensor, retain_graph=True)[0]
                        adjusted_grad_pos = inferstep.step(image_tensor, grad_pos)
                        accumulated_grad_pos += adjusted_grad_pos
                        grad_pos_list.append(grad_pos)
                    # loss = loss/len(positive_classes)
                if len(negative_classes) > 0:
                    for nc in negative_classes:
                        loss_neg = calculate_loss(features, [nc], loss_function)
                        grad_neg = torch.autograd.grad(loss_neg, image_tensor, retain_graph=True)[0]
                        adjusted_grad_neg = inferstep.step(image_tensor, grad_neg)
                        accumulated_grad_neg += adjusted_grad_neg
                        grad_neg_list.append(grad_neg)
                        
                    # print the angle between corresponding grad_pos and grad_neg
                    diff_grad = [grad_pos_list[i] - grad_neg_list[i] for i in range(len(grad_pos_list))]
                    for i in range(len(diff_grad)-1):
                        angle = torch.acos(torch.dot(torch.flatten(diff_grad[i]), torch.flatten(diff_grad[i+1])) / 
                        (torch.norm(diff_grad[i]) * torch.norm(diff_grad[i+1])))
                        angle_deg = torch.rad2deg(angle)
                        angles.append(angle_deg.item())
                        print(f"Angle between diff_grad{i} and diff_grad{i+1}: {angle_deg.item()} degrees")   
                    
                    grad = accumulated_grad_pos - accumulated_grad_neg
                    adjusted_grad = inferstep.step(image_tensor, grad)
                    
                    
                    
            else:
                raise ValueError(f"Loss inference method {loss_infer} not supported")
            

            
            
        
            diffusion_noise = diffusion_noise_ratio * torch.randn_like(image_tensor).cuda()
            if loss_infer == 'GradModulation':
                # image_tensor = inferstep.modulated_project(image_tensor.clone() + adjusted_grad + diffusion_noise*grad_modulation, grad_modulation)
                image_tensor = inferstep.project(image_tensor.clone() + adjusted_grad + diffusion_noise*grad_modulation)

            else:
                image_tensor = inferstep.project(image_tensor.clone() + adjusted_grad + diffusion_noise)    
            
        

        if inference_normalization=='on' and itr == 0:
            denormalized_image_tensor = image_tensor.squeeze(0) * norm_std.view(1, -1, 1, 1) + norm_mean.view(1, -1, 1, 1)
        else:
            denormalized_image_tensor = image_tensor
        
        # contrast = calculate_contrast(denormalized_image_tensor.squeeze(0).detach().cpu().numpy())

        
        if itr in iterations_to_show:
            if inference_config['misc_info']['keep_grads'] and itr>0:
                selected_grad_patterns.append(grad.clone().detach())
            elif inference_config['misc_info']['keep_grads'] and itr==0:
                selected_grad_patterns.append(torch.zeros_like(image_tensor))
            selected_inferred_patterns.append(denormalized_image_tensor.clone().detach())
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).squeeze(-1).squeeze(-1)
            conf, classes = torch.max(probs, 1)
            perceived_categories.append(classes.item())
            confidence_list.append(conf.item())
            
    if inference_config['misc_info']['keep_grads']:
        misc_info_dict['grad_info'] = selected_grad_patterns
        
    if 'positive_classes' in inference_config['misc_info'].keys() or 'negative_classes' in inference_config['misc_info'].keys():
        misc_info_dict['angles_between_gradients'] = angles

    return selected_inferred_patterns, perceived_categories, confidence_list, misc_info_dict



# define a purification function so that it runs generative inference for k times and run the model on the purified image which is the average of the k inferred images
def purification(model_config, image, inference_config, k):
    model = model_config['model']
    norm_mean = model_config['norm_mean']
    norm_std = model_config['norm_std']
    inferred_patterns = []
    grad_patterns = []
    for _ in range(k):
        inferred_patterns_, perceived_categories_, confidence_list_, misc_info_dict_ = generative_inference(model_config, image, inference_config)
        inferred_patterns.append(inferred_patterns_[-1])
        if inference_config['misc_info']['keep_grads']:
            grad_patterns.append(misc_info_dict_['grad_info'][-1])
    purified_image = torch.mean(torch.stack(inferred_patterns), dim=0)
    if inference_config['misc_info']['keep_grads']:
        grad_patterns = torch.mean(torch.stack(grad_patterns), dim=0)
    else:
        grad_patterns = None
    
    
    # apply the same normalization to the purified image as the original image
    if inference_config['recognition_normalization'] == 'on':
        purified_image = purified_image.squeeze(0) * norm_std.view(1, -1, 1, 1) + norm_mean.view(1, -1, 1, 1)
    
    predicted_category = model(purified_image)
    probs = torch.nn.functional.softmax(predicted_category, dim=1).squeeze(-1).squeeze(-1)
    conf, classes = torch.max(probs, 1)
   

    return purified_image, classes, conf, grad_patterns

