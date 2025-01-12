

Kanizsa1 = {'model_args':{'model_arch': 'resnet50',
                            'model_training': 'advrobust_L2_eps_3.00',
                            'dataset': 'imagenet',
                            'epoch_chkpnt': 'full',
                            'norm_mean': tensor([0.4850, 0.4560, 0.4060], device='cuda:0'),
                            'norm_std': tensor([0.2290, 0.2240, 0.2250], device='cuda:0'),
                            'n_classes': 1000},
            
            'inference_args':{'loss_infer': 'ReverseDiffusion',#'IncreaseConfidence',# #'IncreaseConfidence', #'ReverseDiffusion', 
                                'loss_function': 'CE',#'MSE',# #'CE', #'MSE', 
                                'n_itr':101, 
                                'eps': 3, #0.5, 
                                'step_size': 0.5, #3
                                'diffusion_noise_ratio': 0.003,#0.05, 
                                'initial_inference_noise_ratio': 0.1,#0.05, #0.1, 
                                'iterations_to_show': [0, 1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 90, 100],
                                # 'iterations_to_show': [0, 1, 2, 4, 8, 16, 20, 32, 64, 80, 100, 128, 160, 180, 200, 240, 280, 320, 420, 520, 620, 720, 820, 920,],
                                'top_layer': 'all', #'layer4', #'avgpool', #'all',
                                'inference_normalization': 'off',
                                'recognition_normalization': 'off',
                                'misc_info': {'keep_grads': True,}
                    }
}

Texture_defined_Figure_ground = {'model_args':{'model_arch': 'resnet50',
                            'model_training': 'advrobust_L2_eps_3.00',
                            'dataset': 'imagenet',
                            'epoch_chkpnt': 100,
                            'norm_mean': tensor([0.4850, 0.4560, 0.4060], device='cuda:0'),
                            'norm_std': tensor([0.2290, 0.2240, 0.2250], device='cuda:0'),
                            'n_classes': 1000},
                'inference_args':inference_config = {'loss_infer': 'ReverseDiffusion',# #'IncreaseConfidence', #'ReverseDiffusion', 
                                                    'loss_function': 'MSE',#'MSE',# #'CE', #'MSE', 
                                                    'n_itr':101, 
                                                    'eps': 20, #0.5, 
                                                    'step_size': 0.8, #3
                                                    'diffusion_noise_ratio': 0.005,#0.05, 
                                                    'initial_inference_noise_ratio': 0.5,#0.05, #0.1, 
                                                    'iterations_to_show': [0, 1,  4, 8, 16, 20, 30, 40, 60, 80, 90, 100],
                                                    # 'iterations_to_show': [0, 1, 2, 4, 16, 32, 64, 80, 100, 200, 400 ],
                                                    'top_layer': 'layer3', #'layer4', #'avgpool', #'all',
                                                    'inference_normalization': 'off',
                                                    'recognition_normalization': 'off',
                                                    'misc_info': {'keep_grads': True,}
                                                    }

}