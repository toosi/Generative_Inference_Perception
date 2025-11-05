"""Generative inference module for perception and learning."""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F

# URL for the ImageNet labels
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)

# Fetch the label file
response = requests.get(IMAGENET_LABELS_URL)
if response.status_code == 200:
    labels_imagenet = response.json()
else:
    raise RuntimeError("Failed to fetch ImageNet labels")


def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    """Normalize gradient by L2 norm.
    
    Args:
        grad: Gradient tensor to normalize.
        
    Returns:
        Normalized gradient tensor.
    """
    dim = len(grad.shape) - 1
    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * dim))
    scaled_grad = grad / (grad_norm + 1e-10)
    return scaled_grad


class InferStep:
    """Inference step class for gradient-based optimization."""
    
    def __init__(self, orig_image: torch.Tensor, eps: float, step_size: float):
        """Initialize inference step.
        
        Args:
            orig_image: Original image tensor.
            eps: Maximum perturbation bound.
            step_size: Step size for gradient updates.
        """
        self.orig_image = orig_image
        self.eps = eps
        self.step_size = step_size

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project x onto epsilon-ball around original image.
        
        Args:
            x: Image tensor to project.
            
        Returns:
            Projected image tensor.
        """
        diff = x - self.orig_image
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(self.orig_image + diff, 0, 1)
    
    def modulated_project(self, x: torch.Tensor, grad_modulation: float) -> torch.Tensor:
        """Modulated projection with gradient modulation.
        
        Args:
            x: Image tensor to project.
            grad_modulation: Modulation factor for gradient.
            
        Returns:
            Modulated and projected image tensor.
        """
        diff = x - self.orig_image
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(
            self.orig_image * (1 - grad_modulation) + diff * grad_modulation, 0, 1
        )

    def step(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Take a normalized gradient step.
        
        Args:
            x: Current image tensor.
            grad: Gradient tensor.
            
        Returns:
            Scaled gradient step.
        """
        dim = len(x.shape) - 1
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * dim))
        scaled_grad = grad / (grad_norm + 1e-10)
        return scaled_grad * self.step_size

def extract_middle_layers(
    model: nn.Module, layer_index: Union[str, int]
) -> nn.Module:
    """Extract middle layers from a model up to a specified layer index.
    
    Supports ResNet, VGG, and Vision Transformer architectures.
    
    Args:
        model: PyTorch model to extract layers from.
        layer_index: Layer index or 'all' for full model. Can be:
            - 'all': Return full model
            - int: Truncate to specified layer index
            - str: Extract specific layer by name (e.g., 'encoder.layers.encoder_layer_0')
            
    Returns:
        Truncated model up to the specified layer.
        
    Raises:
        ValueError: If layer_index is not found in the model.
    """
    if isinstance(layer_index, str) and layer_index == 'all':
        return model
    
    # Special case for ViT's encoder layers with DataParallel wrapper
    if isinstance(layer_index, str) and layer_index.startswith('encoder.layers.encoder_layer_'):
        try:
            target_layer_idx = int(layer_index.split('_')[-1])
            new_model = copy.deepcopy(model)
            
            # For models wrapped in DataParallel
            if hasattr(new_model, 'module'):
                encoder_layers = nn.Sequential()
                for i in range(target_layer_idx + 1):
                    layer_name = f"encoder_layer_{i}"
                    if hasattr(new_model.module.encoder.layers, layer_name):
                        encoder_layers.add_module(
                            layer_name,
                            getattr(new_model.module.encoder.layers, layer_name)
                        )
                new_model.module.encoder.layers = encoder_layers
                new_model.module.heads = nn.Identity()
                return new_model
            else:
                # Direct model access (not DataParallel)
                encoder_layers = nn.Sequential()
                for i in range(target_layer_idx + 1):
                    layer_name = f"encoder_layer_{i}"
                    if hasattr(new_model.encoder.layers, layer_name):
                        encoder_layers.add_module(
                            layer_name,
                            getattr(new_model.encoder.layers, layer_name)
                        )
                new_model.encoder.layers = encoder_layers
                new_model.heads = nn.Identity()
                return new_model
                
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid ViT layer specification: {layer_index}. Error: {e}"
            )
    
    # Existing handling for ViT whole blocks
    elif hasattr(model, 'blocks') or (
        hasattr(model, 'module') and hasattr(model.module, 'blocks')
    ):
        base_model = model.module if hasattr(model, 'module') else model
        new_model = copy.deepcopy(model)
        base_new_model = new_model.module if hasattr(new_model, 'module') else new_model
        
        # Add the desired number of transformer blocks
        if isinstance(layer_index, int):
            base_new_model.blocks = base_new_model.blocks[:layer_index + 1]
            
        return new_model
    
    else:
        # Original ResNet/VGG handling
        modules = list(model.named_children())
        cutoff_idx = next(
            (i for i, (name, _) in enumerate(modules) if name == str(layer_index)),
            None
        )
        
        if cutoff_idx is not None:
            new_model = nn.Sequential(OrderedDict(modules[:cutoff_idx + 1]))
            return new_model
        else:
            raise ValueError(
                f"Module {layer_index} not found in model. "
                f"Available modules: {[name for name, _ in modules]}"
            )
    
def generate_perlin_noise(image_tensor: torch.Tensor, scale: int = 3) -> torch.Tensor:
    """Generate Perlin noise for image augmentation.
    
    Args:
        image_tensor: Input image tensor.
        scale: Scale factor for noise grid size.
        
    Returns:
        Perlin noise tensor matching image dimensions.
    """
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    
    # Generate random grid of control points
    grid_size = (height // scale, width // scale)
    control_points = np.random.rand(*grid_size).astype(np.float32)
    
    # Resize to match image dimensions using bilinear interpolation
    noise = torch.from_numpy(control_points)
    noise = F.interpolate(
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



def calculate_contrast(image: np.ndarray) -> float:
    """Calculate the RMS (Root Mean Square) contrast of an image.
    
    Args:
        image: A numpy array representing the image (values should be in [0, 1]).
        
    Returns:
        The RMS contrast value.
    """
    image_flat = image.flatten().astype(float)
    mean = np.mean(image_flat)
    std_dev = np.std(image_flat)
    rms_contrast = std_dev / mean if mean != 0 else 0
    return rms_contrast


def powers_of_two_less_than(n: int) -> List[int]:
    """Generate all powers of two less than n.
    
    Args:
        n: Upper bound.
        
    Returns:
        List of powers of two less than n.
    """
    power = 1
    result = []
    while power < n:
        result.append(power)
        power *= 2
    return result


def calculate_loss(
    output_model: torch.Tensor, class_indices: List[int], loss_inference: str
) -> torch.Tensor:
    """Calculate loss for specified class indices.
    
    Args:
        output_model: Model output logits.
        class_indices: List of class indices to compute loss for.
        loss_inference: Loss type ('CE' for CrossEntropy, 'MSE' for Mean Squared Error).
        
    Returns:
        Average loss across specified classes.
    """
    losses = []
    for idx in class_indices:
        target = torch.full((1,), idx, dtype=torch.long, device=output_model.device)
        if loss_inference == 'CE':
            loss = nn.CrossEntropyLoss()(output_model, target)
        elif loss_inference == 'MSE':
            one_hot_target = torch.zeros_like(output_model)
            one_hot_target[0, target] = 1
            loss = nn.MSELoss()(output_model, one_hot_target)
        else:
            raise ValueError(f"Unsupported loss_inference: {loss_inference}")
        losses.append(loss)
    
    return torch.stack(losses).mean()


def generative_inference(
    model_config: Dict,
    image: Union[str, Image.Image],
    inference_config: Dict
) -> Tuple[List[torch.Tensor], List[Union[int, str]], List[float], Dict]:
    """Perform generative inference on an image.
    
    Args:
        model_config: Dictionary containing model configuration:
            - model: PyTorch model
            - input_size: Input image size
            - dataset_model: Dataset name
            - norm_mean: Normalization mean
            - norm_std: Normalization std
            - n_classes: Number of classes
        image: Input image (path string or PIL Image).
        inference_config: Dictionary containing inference configuration:
            - loss_infer: Loss inference method
            - loss_function: Loss function type
            - n_itr: Number of iterations
            - eps: Perturbation bound
            - step_size: Gradient step size
            - diffusion_noise_ratio: Diffusion noise ratio
            - initial_inference_noise_ratio: Initial noise ratio
            - top_layer: Top layer to extract
            - inference_normalization: Normalization flag
            - recognition_normalization: Recognition normalization flag
            - iterations_to_show: Iterations to save
            - misc_info: Additional configuration
            
    Returns:
        Tuple of:
            - selected_inferred_patterns: List of inferred image patterns
            - perceived_categories: List of perceived categories
            - confidence_list: List of confidence scores
            - misc_info_dict: Dictionary with additional information
    """
    
    model = model_config['model']
    input_size = model_config['input_size']
    norm_mean = model_config['norm_mean']
    norm_std = model_config['norm_std']
    n_classes = model_config['n_classes']
    
    # Unpack inference configuration
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
        assert loss_function == 'MSE', (
            "CE loss function is not supported for non-all top layers"
        )
    
    # Check if the image is a path or a PIL image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, torch.Tensor):
        raise ValueError(
            f"Image type {type(image)}, looks like already a transformed tensor"
        )
    
    
  
    if inference_normalization == 'on':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        
    if loss_infer == 'GradModulation':
        grad_modulation = inference_config['misc_info']['grad_modulation']
        image_tensor = transform(image).unsqueeze(0).cuda()
        image_tensor = (
            image_tensor * (1 - grad_modulation) +
            grad_modulation * torch.randn_like(image_tensor)
        )
    else:
        image_tensor = transform(image).unsqueeze(0).cuda()
    image_tensor.requires_grad = True

    # Prepare model
    new_model = extract_middle_layers(model.module, top_layer)
    new_model.eval()
    
    output_original = model(image_tensor)
    try:
        probs_orig = F.softmax(output_original, dim=1).squeeze(-1).squeeze(-1)
        conf_orig, classes_orig = torch.max(probs_orig, 1)
    except Exception:
        pass
    if loss_infer == 'IncreaseConfidence':
        _, least_confident_classes = torch.topk(
            probs_orig, k=int(n_classes / 10), largest=False
        )
    elif loss_infer == 'GradModulation':
        image_sec1 = image_tensor[:, :, 0:112, :].clone()
        image_sec2 = image_tensor[:, :, 112:, 0:112].clone()
        image_sec3 = image_tensor[:, :, 0:112, 0:112].clone()
        image_sec4 = image_tensor[:, :, 0:112, 112:].clone()
        
        image_secs = [image_sec1, image_sec2, image_sec3, image_sec4]
        probs_sec_list = []
        for img in image_secs:
            img = F.interpolate(
                img, size=(224, 224), mode='bilinear', align_corners=True
            )
            output_sec = model(img)
            probs_sec = F.softmax(output_sec, dim=1).squeeze(-1).squeeze(-1)
            probs_sec_list.append(probs_sec)
        probs_secs_mean = torch.mean(torch.stack(probs_sec_list), dim=0)
        _, least_confident_classes = torch.topk(
            probs_secs_mean, k=int(n_classes / 10), largest=False
        )
        
    
    # Noisy image for Reverse Diffusion
    if loss_infer == 'PGDD':
        added_noise = initial_inference_noise_ratio * torch.randn_like(image_tensor).cuda()
        noisy_image_tensor = image_tensor + added_noise
        noisy_features = new_model(noisy_image_tensor)
        
    elif loss_infer == 'CompositionalFusion':
        positive_categories = inference_config['misc_info']['positive_classes']
        negative_categories = inference_config['misc_info']['negative_classes']
        # Find the target indices for the positive and negative categories
        if len(positive_categories) > 0:
            positive_classes = [
                labels_imagenet.index(pc) for pc in positive_categories
            ]
        else:
            positive_classes = []

        if len(negative_categories) > 0:
            negative_classes = [
                labels_imagenet.index(nc) for nc in negative_categories
            ]
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
    probs_list = []
    
    for itr in range(n_itr):
        new_model.zero_grad()
        
        
        if inference_config['misc_info'].get('smooth_inference', False):
            # Compute the smoothed output instead of a single forward pass
            num_samples = inference_config['misc_info'].get('smooth_samples', 100)
            batch_size_smooth = inference_config['misc_info'].get('smooth_batch_size', 1)
            smooth_sigma = inference_config['misc_info'].get('smooth_sigma', 0.25)
            use_multiscale = inference_config['misc_info'].get('smooth_multiscale', False)
            if use_multiscale:
                scales = range(1, 128, 1)
            else:
                scales = [1]
            all_feats = []
            num_remaining = num_samples
            while num_remaining > 0:
                current_batch = min(batch_size_smooth, num_remaining)
                batch = image_tensor.repeat(current_batch, 1, 1, 1)
                if use_multiscale:
                    multi_noise = 0
                    # For each scale, generate noise, upscale it, and sum it
                    for scale in scales:
                        H, W = batch.size(2), batch.size(3)
                        # Calculate new dimensions; ensure minimum size 1x1
                        new_H, new_W = max(1, H // scale), max(1, W // scale)
                        noise_small = torch.randn(
                            current_batch, 3, new_H, new_W, device=batch.device
                        )
                        noise_upsampled = F.interpolate(
                            noise_small, size=(H, W), mode='bilinear', align_corners=False
                        )
                        multi_noise += noise_upsampled
                        noise = multi_noise * smooth_sigma / len(scales)
                else:
                    noise = torch.randn_like(batch) * smooth_sigma
                feat = new_model(batch + noise)
                all_feats.append(feat)
                num_remaining -= current_batch
            # Average the outputs to form the smoothed representation
            features = torch.mean(torch.cat(all_feats, dim=0), dim=0, keepdim=True)
        else:
            new_model.zero_grad()
            features = new_model(image_tensor)
        
        if itr == 0:
            # Don't add priors or diffusion noise to the first iteration
            output = model(image_tensor)
            if isinstance(output, torch.Tensor) and output.size(-1) == n_classes:
                probs = F.softmax(output, dim=1).squeeze(-1).squeeze(-1)
                conf, classes = torch.max(probs, 1)
            else:
                # To handle models like SAM
                probs = 0
                conf = 0
                classes = 'N/A'

        else:
            
            if loss_infer == 'PGDD':
                assert loss_function == 'MSE', "Reverse Diffusion loss function must be MSE"
                loss = torch.nn.functional.mse_loss(features, noisy_features)
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                
            elif loss_infer == 'IncreaseConfidence':
                loss = calculate_loss(features, least_confident_classes[0], loss_function)
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                
                
            # for resolving occlusions
            elif loss_infer == 'GradModulation':
                grad_modulation = inference_config['misc_info']['grad_modulation']
                loss = calculate_loss(
                    features, least_confident_classes[0], loss_function
                )
                grad = torch.autograd.grad(loss, image_tensor)[0]
                adjusted_grad = inferstep.step(image_tensor, grad)
                
                
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
                        
                    # Print the angle between corresponding grad_pos and grad_neg
                    diff_grad = [
                        grad_pos_list[i] - grad_neg_list[i]
                        for i in range(len(grad_pos_list))
                    ]
                    for i in range(len(diff_grad) - 1):
                        dot_product = torch.dot(
                            torch.flatten(diff_grad[i]),
                            torch.flatten(diff_grad[i + 1])
                        )
                        norms = torch.norm(diff_grad[i]) * torch.norm(diff_grad[i + 1])
                        angle = torch.acos(dot_product / norms)
                        angle_deg = torch.rad2deg(angle)
                        angles.append(angle_deg.item())
                        print(
                            f"Angle between diff_grad{i} and diff_grad{i+1}: "
                            f"{angle_deg.item()} degrees"
                        )   
                    
                    grad = accumulated_grad_pos - accumulated_grad_neg
                    adjusted_grad = inferstep.step(image_tensor, grad)
                    
                    
                    
            else:
                raise ValueError(f"Loss inference method {loss_infer} not supported")
            

            
            
            diffusion_noise = diffusion_noise_ratio * torch.randn_like(image_tensor).cuda()
            if loss_infer == 'GradModulation':
                image_tensor = inferstep.project(
                    image_tensor.clone() +
                    adjusted_grad * grad_modulation +
                    diffusion_noise * grad_modulation
                )
            else:
                image_tensor = inferstep.project(
                    image_tensor.clone() + adjusted_grad + diffusion_noise
                )
            
        

        if inference_normalization == 'on' and itr == 0:
            denormalized_image_tensor = (
                image_tensor.squeeze(0) * norm_std.view(1, -1, 1, 1) +
                norm_mean.view(1, -1, 1, 1)
            )
        elif inference_normalization == 'off' and itr == 0:
            denormalized_image_tensor = image_tensor
        else:
            denormalized_image_tensor = image_tensor
        
        if itr in iterations_to_show:
            if inference_config['misc_info']['keep_grads'] and itr > 0:
                selected_grad_patterns.append(grad.clone().detach())
            elif inference_config['misc_info']['keep_grads'] and itr == 0:
                selected_grad_patterns.append(torch.zeros_like(image_tensor))
            selected_inferred_patterns.append(denormalized_image_tensor.clone().detach())
            output = model(image_tensor)
            if isinstance(output, torch.Tensor) and output.size(-1) == n_classes:
                probs = F.softmax(output, dim=1).squeeze(-1).squeeze(-1)
                conf, classes = torch.max(probs, 1)
                classes = classes.item()
                conf = conf.item()
            else:
                probs = 0
                conf = 0
                classes = 'N/A'
            perceived_categories.append(classes)
            confidence_list.append(conf)
            probs_list.append(probs)
            
    if inference_config['misc_info']['keep_grads']:
        misc_info_dict['grad_info'] = selected_grad_patterns
        
    if ('positive_classes' in inference_config['misc_info'].keys() or
            'negative_classes' in inference_config['misc_info'].keys()):
        misc_info_dict['angles_between_gradients'] = angles
    misc_info_dict['probs_list'] = probs_list

    return selected_inferred_patterns, perceived_categories, confidence_list, misc_info_dict


def purification(
    model_config: Dict,
    image: Union[str, Image.Image],
    inference_config: Dict,
    k: int
) -> Tuple[torch.Tensor, Union[int, str], float, Optional[torch.Tensor]]:
    """Run generative inference k times and return averaged purified image.
    
    Runs generative inference k times and returns the model prediction on the
    averaged inferred image.
    
    Args:
        model_config: Dictionary containing model configuration.
        image: Input image (path string or PIL Image).
        inference_config: Dictionary containing inference configuration.
        k: Number of inference runs to average.
        
    Returns:
        Tuple of:
            - purified_image: Averaged inferred image.
            - classes: Predicted class.
            - conf: Confidence score.
            - grad_patterns: Gradient patterns (if keep_grads is True).
    """
    model = model_config['model']
    norm_mean = model_config['norm_mean']
    norm_std = model_config['norm_std']
    inferred_patterns = []
    grad_patterns = []
    
    for _ in range(k):
        (inferred_patterns_, perceived_categories_, confidence_list_,
         misc_info_dict_) = generative_inference(
            model_config, image, inference_config
        )
        inferred_patterns.append(inferred_patterns_[-1])
        if inference_config['misc_info']['keep_grads']:
            grad_patterns.append(misc_info_dict_['grad_info'][-1])
    
    purified_image = torch.mean(torch.stack(inferred_patterns), dim=0)
    if inference_config['misc_info']['keep_grads']:
        grad_patterns = torch.mean(torch.stack(grad_patterns), dim=0)
    else:
        grad_patterns = None
    
    # Apply the same normalization to the purified image as the original image
    if inference_config['recognition_normalization'] == 'on':
        purified_image = (
            purified_image.squeeze(0) * norm_std.view(1, -1, 1, 1) +
            norm_mean.view(1, -1, 1, 1)
        )
    
    predicted_category = model(purified_image)
    probs = F.softmax(predicted_category, dim=1).squeeze(-1).squeeze(-1)
    conf, classes = torch.max(probs, 1)

    return purified_image, classes, conf, grad_patterns

