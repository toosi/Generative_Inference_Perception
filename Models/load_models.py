"""Load models from checkpoints.

This script loads robust models from checkpoints.
See: https://github.com/standardgalactic/robust-models-transfer
"""

import os
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import timm
from robustness import model_utils
from torchvision import models as torchvision_models

from Models import hash_checkpoints

path_codes = os.path.abspath(os.getcwd())
path_checkpoints = os.path.join(path_codes, 'Models/models_checkpoints')
path_checkpoints_faces = (
    '/home/tahereh/engram/users/Tahereh/Codes/Perceptually_Aligned_Gradients/'
    'Training/TrainedModels'
)

dict_hash_resent50_vggface2 = {
    'advrobust_L2_eps_0.50': '3140bf7b-21a2-41dd-9ace-9407fcfc2137',
    'advrobust_L2_eps_1.00': '0bad943b-4373-4bba-998a-95493dd2bf51',
    'advrobust_L2_eps_3.00': 'c91917b7-660b-4c04-97f4-993be74e6eb8',
}


def normalize_fn(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torchvision.functional.normalize.
    
    Args:
        tensor: Input tensor (assumes color channel at dim=1).
        mean: Mean values.
        std: Standard deviation values.
        
    Returns:
        Normalized tensor.
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    """Normalization layer by channel mean and std."""
    
    def __init__(self, mean: Union[torch.Tensor, list], std: Union[torch.Tensor, list]):
        """Initialize normalization layer.
        
        Args:
            mean: Mean values per channel.
            std: Standard deviation values per channel.
        """
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tensor: Input tensor.
            
        Returns:
            Normalized tensor.
        """
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'mean={self.mean}, std={self.std}'


dummy_path = '.'


def load_models(args) -> Tuple[nn.Module, object]:
    """Load models from checkpoints.
    
    Args:
        args: Configuration object with:
            - dataset: Dataset name ('imagenet' or 'vggface2')
            - model_arch: Model architecture
            - model_training: Training configuration string
            - epoch_chkpnt: Checkpoint epoch ('full' or int)
            
    Returns:
        Tuple of (model, args) with updated args containing load_path.
        
    Raises:
        ValueError: If dataset or model architecture is not supported.
    """
    # Hard-coded dataset, architecture, batch size, workers
    from robustness.datasets import ImageNet
    
    if args.dataset == 'imagenet':
        ds = ImageNet(os.path.join(dummy_path, 'imagenet'))
        num_classes = 1000
    elif args.dataset == 'vggface2':
        import sys
        sys.path.append(os.path.join(path_codes, 'Datasets'))
        from VGGface2 import Vggface2forRobustness
        num_classes = 500
        n_train_per_class = 200
        n_val_per_class = 40
        ds = Vggface2forRobustness(
            num_classes=num_classes,
            n_train_per_class=n_train_per_class,
            n_val_per_class=n_val_per_class
        )
    else:
        raise ValueError("The dataset is not supported")
    


    

    if args.dataset == 'imagenet':
        if args.model_arch == 'resnet50':
            if args.epoch_chkpnt == 'full' or isinstance(args.epoch_chkpnt, int):
                args.eps = float(args.model_training.split('_')[-1])
                if args.epoch_chkpnt == 'full':
                    load_path = (
                        path_checkpoints +
                        f'/{args.model_arch}_{args.dataset}_L2_eps_{args.eps:.2f}'
                        '_checkpoint.pt.best'
                    )
                print(f"Loading model from {load_path}")
                if isinstance(args.epoch_chkpnt, int):
                    assert int(args.epoch_chkpnt) % 2 == 0, (
                        "The epoch number should be even"
                    )
                
                model, _ = model_utils.make_and_restore_model(
                    arch='resnet50', dataset=ds, resume_path=load_path
                )
                print(f"****** Loaded model from {load_path}")
            
        
    elif args.dataset == 'vggface2':
        args.eps = float(args.model_training.split('_')[-1])
        if args.epoch_chkpnt == 'full':
            load_path = (
                path_checkpoints_faces +
                f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/'
                f'{dict_hash_resent50_vggface2[args.model_training]}/'
                'checkpoint.pt.best'
            )
        else:
            assert (
                int(args.epoch_chkpnt) < 199 and
                int(args.epoch_chkpnt) % 2 == 0
            ), "The epoch number should be less than 199 and even"
            load_path = (
                path_checkpoints_faces +
                f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/'
                f'{dict_hash_resent50_vggface2[args.model_training]}/'
                f'{args.epoch_chkpnt}_checkpoint.pt'
            )

        if args.model_arch == 'vgg16':
            model_arch = torchvision_models.vgg16(pretrained=False)
            model_arch.classifier[-1] = nn.Linear(
                in_features=model_arch.classifier[-1].in_features,
                out_features=100
            )
        else:
            model_arch = timm.create_model(
                args.model_arch, num_classes=500, pretrained=False
            )

        print(
            f"****** Loading model from {load_path} "
            "after adjusting the model architecture"
        )
        try:
            model = model_utils.make_and_restore_model(
                arch=model_arch, dataset=ds, resume_path=load_path,
                add_custom_forward=True
            )
            model = model[0].model
        except Exception:
            # Wrap the model in a model wrapper with model name
            checkpoint = torch.load(load_path)
            checkpoint_model = checkpoint['model']
            # Drop model. from the beginning of the key
            checkpoint_model = {
                k.replace('model', 'model.model'): v
                for k, v in checkpoint_model.items()
            }
            checkpoint['model'] = checkpoint_model
            torch.save(checkpoint, 'tmp.pth')
            
            model = model_utils.make_and_restore_model(
                arch=model_arch, dataset=ds, resume_path='tmp.pth',
                add_custom_forward=True
            )
            model = model[0].model
            
    
    
    
    else:
        raise ValueError("The model architecture is not supported")

    model = model.model
    _=model.eval()
    _=model.cuda()
    args.load_path = load_path

    return model, args


