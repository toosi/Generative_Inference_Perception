
# This script loads the models from the checkpoints.
# checkout https://github.com/standardgalactic/robust-models-transfer?tab=readme-ov-file#download-our-robust-imagenet-models


# import socket
# import os
# path_codes = '/content/drive/MyDrive/tt2684@columbia.edu 2025-06-27 15:21/Postdoc Projects/Publications/Generative Inference/Generative_Inference_Perception'

# import sys
# sys.path.append(path_codes)
# # sys.path.append(os.path.join(path_prefix, 'Codes/Perceptually_Aligned_Gradients/Utils'))
# print(sys.path)



from robustness import model_utils, datasets, train, defaults

# _, val_loader_robustness = ds.make_loaders(batch_size=args.batch_size ,val_batch_size=args.batch_size, workers=0,  shuffle_val=False)
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch 
import torch.nn as nn
from torchvision import models as torchvision_models
import os

path_codes = os.path.abspath(os.getcwd())
print(path_codes)

path_checkpoints = os.path.join(path_codes, 'Models/models_checkpoints')
path_checkpoints_faces = '/home/tahereh/engram/users/Tahereh/Codes/Perceptually_Aligned_Gradients/Training/TrainedModels'

dict_hash_resent50_vggface2 = {
                      'advrobust_L2_eps_0.50': '3140bf7b-21a2-41dd-9ace-9407fcfc2137',
                      'advrobust_L2_eps_1.00': '0bad943b-4373-4bba-998a-95493dd2bf51',
                      'advrobust_L2_eps_3.00': 'c91917b7-660b-4c04-97f4-993be74e6eb8',}

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

## define models 
import os
import timm
import argparse

import sys
from Models import hash_checkpoints

import torch
from typing import Dict, Union
from collections import OrderedDict

# # these models are robustly trained on imagenet
# other_CNN_models = ['resnet18', 'densenet','mnasnet', 'mobilenet-v3', 'wide-resnet50-2', 'vgg16-bn','shufflenet','resnext50-32x4d']


dummy_path ='.'

def load_models(args):
    print(args)
  
    ## Hard-coded dataset, architecture, batch size, workers
    from robustness.datasets import ImageNet
    if args.dataset == 'imagenet':
        from robustness.datasets import ImageNet
        ds = ImageNet(os.path.join(dummy_path ,'imagenet'))
        num_classes = 1000
    
    elif args.dataset == 'vggface2':
        import sys
        sys.path.append(os.path.join(path_codes,'Datasets'))
    
        from VGGface2 import Vggface2forRobustness
        # Hard-coded dataset, architecture, batch size, workers
        num_classes = 500
        n_train_per_class = 200
        n_val_per_class = 40
        ds = Vggface2forRobustness(num_classes=num_classes,
                                    n_train_per_class=n_train_per_class,
                                    n_val_per_class=n_val_per_class)
        

    else:
        raise ValueError("The dataset is not supported")
    


    

    if args.dataset == 'imagenet': 
        
        
        if (args.model_arch == 'resnet50'): 
        
                
            if (args.epoch_chkpnt == 'full' or isinstance(args.epoch_chkpnt, int)):
                
                # args.eps = float(args.model_training.split('_')[-1])
                # load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'
                
                print(f"Loading model from imagenet")
                args.eps = float(args.model_training.split('_')[-1])
                if args.epoch_chkpnt == 'full':
                    load_path = path_checkpoints+ f'/{args.model_arch}_{args.dataset}_L2_eps_{args.eps:.2f}_checkpoint.pt.best'
                # else:
                #     assert (int(args.epoch_chkpnt) < 199) and int(args.epoch_chkpnt)%2==0, "The epoch number should be less than 199 and even"

                #     load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'

                
                print(f"Loading model from {load_path}")
                if isinstance(args.epoch_chkpnt, int):
                    assert int(args.epoch_chkpnt)%2==0, "The epoch number should be even"
                
                model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=load_path,)        
                print(f"****** Loaded model from {load_path}")
            
        
    elif (args.dataset == 'vggface2'):
        
        args.eps = float(args.model_training.split('_')[-1])
        if args.epoch_chkpnt == 'full':
            load_path = path_checkpoints_faces+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash_resent50_vggface2[args.model_training]}/checkpoint.pt.best'
        else:
            assert (int(args.epoch_chkpnt) < 199) and int(args.epoch_chkpnt)%2==0, "The epoch number should be less than 199 and even"
            load_path = path_checkpoints_faces+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash_resent50_vggface2[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'
            print('load_path:', load_path)

        if args.model_arch == 'vgg16':
            model_arch = torchvision_models.vgg16(pretrained=False)
            model_arch.classifier[-1] = torch.nn.Linear(in_features=model_arch.classifier[-1].in_features, out_features=100) 
        else:
        

              model_arch = timm.create_model(args.model_arch, num_classes=500, pretrained=False)
  
        print(f"****** Loading model from {load_path} after adjusting the model architecture")
        try: 
            model = model_utils.make_and_restore_model(arch=model_arch, dataset=ds,
                                                        resume_path=load_path,add_custom_forward=True,)
            
            print(len(model))
            print(model[0]) 
            model = model[0].model
            
        except:
            # wrap the model in a model wrapper with model name
            # model_arch = nn.DataParallel(model_arch)
            
            checkpoint = torch.load(load_path)
            checkpoint_model = checkpoint['model']
            print(checkpoint_model.keys())
            # drop model. from the begining of the key
            checkpoint_model = {k.replace('model','model.model'): v for k, v in checkpoint_model.items()}
            
            print(checkpoint_model.keys())
            checkpoint['model'] = checkpoint_model
            torch.save(checkpoint, 'tmp.pth')
        
            model = model_utils.make_and_restore_model(arch=model_arch, dataset=ds,
                                                        resume_path='tmp.pth',add_custom_forward=True,)
            
            print(len(model))
            print(model[0]) 
            model = model[0].model
            
    
    
    
    else:
        raise ValueError("The model architecture is not supported")

    model = model.model
    _= model.eval()
    _= model.cuda()
    args.load_path = load_path

    return model, args


