
# This script loads the models from the checkpoints.
# checkout https://github.com/standardgalactic/robust-models-transfer?tab=readme-ov-file#download-our-robust-imagenet-models


import socket
import os
hostname = socket.gethostname()

if hostname.startswith('ax'): 
    print('The kernel is running on an Axon server {}'.format(hostname))
    path_prefix = '/mnt/smb/locker/miller-locker/users/Tahereh'
    path_prefix_data = '/home/tt2684/Research/Data'
elif hostname == 'demo':
    print("Kernel running on local computer 'demo'.")
    path_prefix = '/home/tahereh/engram/users/Tahereh'
    path_prefix_data = '/home/tahereh/engram/users/Tahereh/Research/Data'

import sys
sys.path.append(os.path.join(path_prefix, 'Codes/Perceptually_Aligned_Gradients'))
# sys.path.append(os.path.join(path_prefix, 'Codes/Perceptually_Aligned_Gradients/Utils'))
print(sys.path)



from robustness import model_utils, datasets, train, defaults

# _, val_loader_robustness = ds.make_loaders(batch_size=args.batch_size ,val_batch_size=args.batch_size, workers=0,  shuffle_val=False)
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch 
import torch.nn as nn

dict_chkpts_urls = {'advrobust_resnet50': 'http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_models/ours/examples/adversarial_training/model_best.pth.tar',
                    'advrobust_vgg16':'http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vgg16_ep4.pth',
                    'advrobust_vit_base_patch32_224':'http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/advtrain_models/advtrain_vit_base_patch32_224_ep4.pth'}


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
    
# dict_hash_CIFAR10 = {'advrobust_L2_eps_0.25': '09dbbb67-a704-443b-ad53-9ab9f017f5b8',
#                 'advrobust_L2_eps_0.50': '7abb1f1d-cd46-4687-9073-3d5944e19f7c',
#                 'advrobust_L2_eps_1.00': 'bdd6e627-8f10-4417-8bcd-5af9de79f417',
#                 'advrobust_L2_eps_0.00': 'dd858fd2-ce15-4f26-b3d9-9d93c3479b53'}

# dict_hash_MNIST = {

# 'advrobust_L2_eps_0.00': '1795c646-05f2-495c-bc58-0963387c7e60', #107
# 'advrobust_L2_eps_0.10': '962d351d-d87e-47ba-9dd7-bd58a15af686', #107
# 'advrobust_L2_eps_0.25': '1d073cb8-4e9f-4914-900b-29eb8a8996c0', #107
# 'advrobust_L2_eps_0.50': '4a3347b7-7746-4903-9982-2d4d66bd80c1', #107
# 'advrobust_L2_eps_1.00': 'd0e3c4dc-2e5d-45ec-9a03-876fefcf02cd', #107
# 'advrobust_L2_eps_2.00': 'aaeabb2b-3456-4583-9c2d-98588161ac6e', #107
# 'advrobust_L2_eps_3.00': '0cd9620f-dbb8-4dc4-9253-cc577409f0cd', #107
# 'advrobust_L2_eps_4.00': 'd1ecdff7-3c37-4d24-9b77-61bda413b6fc', #107
# 'advrobust_L2_eps_5.00': 'c30a3ed5-f4e9-4c7a-8f53-a06a1f477d0d', #107
# 'advrobust_L2_eps_6.00' :'a40d52cc-c3a5-4c6d-a49c-87c6e550b97d', #107

# # advrobust_L2_eps_5.00 e29c9509-52b1-449d-b952-3efffd81704d 107
# # advrobust_L2_eps_4.00 2e2d68c9-32af-4745-b9d7-181dc4b1944f 107
# # advrobust_L2_eps_6.00 7ae89da6-5f6a-4a21-ace0-61cc5b9d95a6 ,107
# # advrobust_L2_eps_2.00: 'fc597c7a-6610-4f34-af3d-f0fc93c1be0e', 107
# # advrobust_L2_eps_2.00: '7464d423-cb82-4cd9-94df-7ad8d409b15c', 107
# # advrobust_L2_eps_1.00 b53607da-57aa-4944-aa8b-6f93f94d73bc 107
# # advrobust_L2_eps_1.00 1177ea9d-22d6-4181-9bb2-eb0d22ff8f59 107

# }

## define models 
import os
import timm
import argparse

import sys
sys.path.append(os.path.join(path_prefix, 'Codes/Perceptually_Aligned_Gradients'))
from Models import hash_checkpoints


def load_models(args):
    if isinstance(args, dict):
        print("args is a dictionary")
        args = argparse.Namespace(**args)
    else:
        print("args is not a dictionary")
        pass
    # if args.dataset != 'imagenet':
    #     print("add imagenet models to the hash_checkpoints.py")
    dict_hash = hash_checkpoints.get_dict_hash(args.dataset, args.model_arch)
    
    ## Hard-coded dataset, architecture, batch size, workers
    from robustness.datasets import ImageNet
    if args.dataset == 'imagenet':
        from robustness.datasets import ImageNet
        ds = ImageNet(os.path.join(path_prefix_data ,'imagenet'))
        num_classes = 1000
    
    elif args.dataset == 'vggface2':
        import sys
        sys.path.append(os.path.join(path_prefix,'Codes/Perceptually_Aligned_Gradients/Datasets'))
    
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
    

    if (args.model_training == 'standard') and (args.dataset == 'imagenet'):
        model, _ = model_utils.make_and_restore_model(arch=args.model_arch, dataset=ds,
                                            pytorch_pretrained=True,)

    ## load from checkpoints

    else:
        

        path_dir = os.path.join(path_prefix, 'Codes/Perceptually_Aligned_Gradients') #os.path.dirname(os.path.realpath(__file__))
        path_checkpoints = os.path.join(path_dir, 'Training/TrainedModels')

        if args.model_arch == 'resnet50': 
            
            
            if (args.dataset == 'imagenet') and (args.epoch_chkpnt == 'full'):
                if args.model_training == 'advrobust_L2_eps_3.00':
                    args.eps = float(args.model_training.split('_')[-1])

                    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=path_checkpoints+ f'/madry_robust/imagenet/L2/imagenet_l2_3_0.pt',)
                    load_path = path_checkpoints+ f'/madry_robust/imagenet/L2/imagenet_l2_3_0.pt'
                    print("loading model from: ", load_path)
                
                elif args.model_training == 'advrobust_Linf_eps_4.00':
                    args.eps = float(args.model_training.split('_')[-1])

                    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=path_checkpoints+ f'/madry_robust/imagenet/Linf/imagenet_linf_4.pt',)
                    load_path = path_checkpoints+ f'/madry_robust/imagenet/Linf/imagenet_linf_4.pt'
                    print("loading model from: ", load_path)
                    
                elif args.model_training == 'advrobust_Linf_eps_8.00':
                    args.eps = float(args.model_training.split('_')[-1])

                    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=path_checkpoints+ f'/madry_robust/imagenet/Linf/imagenet_linf_8.pt',)
                    load_path = path_checkpoints+ f'/madry_robust/imagenet/Linf/imagenet_linf_8.pt'
                    print("loading model from: ", load_path)
                    
                elif 'gaussrobust' in  args.model_training :
                    args.eps = float(args.model_training.split('_')[-1])

                    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=path_checkpoints+ f'/random_smoothing_robust/imagenet/resnet50/noise_{args.eps:.2f}/checkpoint.pth.tar',)
                    load_path = path_checkpoints+ f'/random_smoothing_robust/imagenet/resnet50/noise_{args.eps:.2f}/checkpoint.pth.tar'
                    print("loading model from: ", load_path)
                    
                    
            elif (args.dataset == 'imagenet') and (args.epoch_chkpnt != 'full'):
                
                args.eps = float(args.model_training.split('_')[-1])
                load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'
                
                print(f"Loading model from {load_path}")
                assert int(args.epoch_chkpnt)%2==0, "The epoch number should be even"
                
                model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=load_path,)        
                print(f"****** Loaded model from {load_path}")
                
            elif (args.dataset == 'Places365') and (args.epoch_chkpnt != 'full'):
                print(f"Loading model from Places365")
                args.eps = float(args.model_training.split('_')[-1])
                assert (int(args.epoch_chkpnt) < 103) and int(args.epoch_chkpnt)%2==0, "The epoch number should be less than 103 and even"
                load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'
                model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=load_path,)        
                print(f"****** Loading model from {load_path}")
                
            elif (args.dataset == 'imagenetvggface2') and (args.epoch_chkpnt != 'full'):
                print(f"Loading model from imagenetvggface2")
                args.eps = float(args.model_training.split('_')[-1])
                load_dir = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}'
                files = os.listdir(load_dir)
                num_checkpoints = len([f for f in files if f.endswith('_checkpoint.pt')])*2
                
                assert (int(args.epoch_chkpnt) < num_checkpoints) and int(args.epoch_chkpnt)%2==0, f"The epoch number should be less than {num_checkpoints} and even"
                load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'
                model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                            resume_path=load_path,)        
                print(f"****** Loading model from {load_path}")
                
            elif (args.dataset == 'vggface2') and (args.epoch_chkpnt != 'full'):
                
                args.eps = float(args.model_training.split('_')[-1])
                if args.epoch_chkpnt == 'full':
                    load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/checkpoint.pt.best'
                else:
                    assert (int(args.epoch_chkpnt) < 199) and int(args.epoch_chkpnt)%2==0, "The epoch number should be less than 199 and even"

                    load_path = path_checkpoints+ f'/train_{args.model_arch}_{args.dataset}_eps_{args.eps:.2f}/{dict_hash[args.model_training]}/{args.epoch_chkpnt}_checkpoint.pt'

   
                model_arch = timm.create_model('resnet50', num_classes=500, pretrained=False)
                #     # wrap the model in a model wrapper with model name
                #     model_arch = nn.DataParallel(model_arch)
                    
                #     checkpoint = torch.load(load_path)
                #     checkpoint_model = checkpoint['model']
                #     # drop model. from the begining of the key
                #     checkpoint_model = {k.replace('module','model'): v for k, v in checkpoint_model.items() if 'model.model' in k}
                    
                #     checkpoint['model'] = checkpoint_model
                #     torch.save(checkpoint, load_path)
                    
                model = model_utils.make_and_restore_model(arch=model_arch, dataset=ds,
                                                            resume_path=load_path,add_custom_forward=True,)
                print(f"****** Loading model from {load_path} after adjusting the model architecture")
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

# bc the model is wrapped in a model wrapper (attacker)



if __name__ == '__main__':
    # put args in a dictionary
    import argparse
    import timm
    argparser = argparse.ArgumentParser(description='load FF models')
    argparser.add_argument('--model_arch', type=str, help='The architecture of the model',
                        choices=['resnet50']+ timm.list_models())
    # look into transformers, etc
    argparser.add_argument('--model_training', type=str, required=True, help='The model to evaluate',
                        choices=['standard',
                                    'advrobust_easyrobust',
                                    'advrobust_L2_eps_3.00','advrobust_L2_eps_1.00',
                                    'advrobust_L2_eps_0.00', 'advrobust_L2_eps_0.25','advrobust_L2_eps_0.50','advrobust_Linf_eps_8.00'])
    argparser.add_argument('--epoch_chkpnt', type=str, default='full', help='The epoch of the model to evaluate')

    argparser.add_argument('--dataset', type=str, required=True, help='trained on dataset',
                        choices=['imagenet','CIFAR10','Places365','MNIST'])



    args = argparser.parse_args()
    if args.model_training == 'standard':
        args.eps = 0.00

    model, model_args = load_models(args=args)