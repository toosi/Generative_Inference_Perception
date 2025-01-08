import numpy as np
import torch
import torchvision
import torch.nn as nn
from Models.model_utils import extract_middle_layers

class Inference:
    def __init__(self, model, model_args):
        """
        Initialize the Inference class.
        
        Args:
            model: The model to use for inference
            model_args (dict): Model configuration arguments
        """
        self.model = model
        self.model_args = model_args
        self.setup_normalization()

    def setup_normalization(self):
        """Set up normalization parameters based on dataset."""
        if self.model_args['dataset'] in ['imagenet', 'imagenetvggface2']:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif self.model_args['dataset'] == 'vggface2':
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

    def get_transform(self, normalization):
        """Create the transform pipeline."""
        if normalization == 'on':
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
            ])

    def project(self, x, orig_image, eps, mask=None):
        """Project the image within epsilon ball around original image."""
        diff = x - orig_image
        diff = torch.clamp(diff, -eps, eps)
        if mask is not None:
            diff = diff * mask
        return torch.clamp(orig_image + diff, 0, 1)

    def step(self, x, grad, step_size):
        """Compute the step based on gradient."""
        l = len(x.shape) - 1
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_grad = grad / (grad_norm + 1e-10)
        return scaled_grad * step_size

    def random_perturb(self, x, eps):
        """Add random perturbation to image."""
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + eps * rp / (rp_norm + 1e-10), 0, 1)

    def generative_inference(self, image, GenInargs):
        """
        Perform generative inference on the input image.
        
        Args:
            image: Input image to perform inference on
            GenInargs: Configuration arguments for generative inference
            
        Returns:
            tuple: (selected_inferred_patterns, perceived_categories, confidence_list, misc_info_dict)
        """
        # Extract arguments from GenInargs
        loss_infer = GenInargs['loss_infer']
        loss_function = GenInargs['loss_function']
        n_itr = GenInargs['n_itr']
        step_size = GenInargs['step_size']
        eps = GenInargs['eps']
        diffusion_noise = GenInargs['diffusion_noise_ratio']
        initial_noise = GenInargs['initial_inference_noise_ratio']
        iterations_to_show = GenInargs['iterations_to_show']
        inference_normalization = GenInargs['inference_normalization']
        keep_grads = GenInargs['misc_info']['keep_grads']
        
        # Transform image
        transform = self.get_transform(inference_normalization)
        image = transform(image).unsqueeze(0).cuda()
        
        # Setup inference
        image = image.detach().clone()
        orig_image = image.clone()
        image.requires_grad = True
        
        # Get original predictions
        output_original = self.model(image)
        probs_orig = torch.nn.functional.softmax(output_original, dim=1).squeeze(-1).squeeze(-1)
        conf_orig, classes_orig = torch.max(probs_orig, 1)
        _, least_likely_indices = torch.topk(probs_orig, k=2, largest=False)
        
        selected_inferred_patterns = []
        perceived_categories = []
        confidence_list = []
        grads_list = []
        
        # Add initial noise if specified
        if initial_noise > 0:
            noise = initial_noise * torch.randn_like(image).cuda()
            image = torch.clamp(image + noise, 0, 1)
        
        # Main inference loop
        for itr in range(n_itr):
            self.model.zero_grad()
            output_model = self.model(image)
            
            # Calculate losses for least likely classes
            losses = []
            for idx in least_likely_indices[0]:
                target = torch.full((1,), idx, dtype=torch.long, device=output_model.device)
                if loss_function == 'CE':
                    loss = nn.CrossEntropyLoss()(output_model, target)
                elif loss_function == 'MSE':
                    one_hot_target = torch.zeros_like(output_model)
                    one_hot_target[0, target] = 1
                    loss = nn.MSELoss()(output_model, one_hot_target)
                losses.append(loss)
            
            # Calculate average loss and update
            loss = torch.stack(losses).mean()
            probs = torch.nn.functional.softmax(output_model, dim=1).squeeze(-1).squeeze(-1)
            conf, classes = torch.max(probs, 1)
            
            if keep_grads:
                grad = torch.autograd.grad(loss, image, retain_graph=True)[0]
            else:
                grad = torch.autograd.grad(loss, image)[0]
            
            # Apply updates
            adjusted_grad = self.step(image, grad, step_size)
            noise = diffusion_noise * torch.randn_like(image).cuda()
            image = self.project(image.clone() + adjusted_grad + noise, orig_image, eps)
            
            perceived_categories.append(classes.item())
            confidence_list.append(conf.item())
            
            # Save at specified iterations
            if itr in iterations_to_show:
                selected_inferred_patterns.append(image.clone())  # Store just the tensor
                if keep_grads:
                    grads_list.append(grad.clone())
        
        misc_info_dict = {'grads': grads_list} if keep_grads else {}
        
        return selected_inferred_patterns, perceived_categories, confidence_list, misc_info_dict