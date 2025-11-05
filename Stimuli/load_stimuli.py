"""Load stimuli images for experiments."""

import os
import socket
from typing import Tuple

import numpy as np
import requests
from PIL import Image
from torchvision import transforms

hostname = socket.gethostname()
if hostname.startswith('ax'):
    print(f'The kernel is running on an Axon server {hostname}')
    path_prefix = '/mnt/smb/locker/miller-locker/users/Tahereh'
    path_prefix_data = '/home/tt2684/Research/Data'
    path_prefix_codes = '/mnt/smb/locker/miller-locker/users/Tahereh/Codes'
elif hostname == 'demo':
    print("Kernel running on local computer 'demo'.")
    path_prefix = '/home/tahereh/engram/users/Tahereh'
    path_prefix_data = '/home/tahereh/engram/users/Tahereh/Research/Data'
    path_prefix_codes = '/home/tahereh/engram/users/Tahereh/Codes'

Imagedir_path = os.path.join(
    path_prefix_codes, 'Public_Codes/Generative_Inference/Stimuli'
)

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
    raise RuntimeError("Failed to fetch labels")

labels_imagenetvggface2 = labels_imagenet + ['face']




from PIL import ImageOps


class ImageLoader:
    """Loader for experiment stimuli images."""
    
    def __init__(self, model_dataset: str):
        """Initialize image loader.
        
        Args:
            model_dataset: Dataset name ('cifar', 'imagenet', 'vggface2', etc.).
            
        Raises:
            ValueError: If dataset is not supported.
        """
        self.Imagedir_path = Imagedir_path
        self.model_dataset = model_dataset

        # Define CIFAR and ImageNet transformations
        if model_dataset == 'cifar':
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),  # CIFAR-10 normalization
            ])
        elif model_dataset in ['imagenet', 'imagenetvggface2', 'Places365']:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ])
        elif model_dataset == 'vggface2':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            raise ValueError(
                "Unsupported dataset. Please choose either 'cifar' or 'imagenet' like."
            )

    def load_image(self, image_name: str) -> Tuple[Image.Image, str]:
        """Load image by name.
        
        Args:
            image_name: Name of the image to load.
            
        Returns:
            Tuple of (PIL Image, colormap name).
            
        Raises:
            ValueError: If image name is not supported.
        """
        if image_name == 'KanizsaSq':   
            image_original = Image.open(os.path.join(self.Imagedir_path, "Kanizsa_square.jpg")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'KanizsaTri':   
            image_original = Image.open(os.path.join(self.Imagedir_path, "KanizsaTri.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'FaceVase':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
            
        elif image_name == 'FaceVaseBlack':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_black.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
        elif image_name == 'FaceVaseWhite':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_white.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
        elif image_name == 'FaceVasecontrolrect':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_controlrectangle.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
        
        elif image_name == 'FaceVasecontroloval':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_controloval.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
            
            
        elif image_name == 'FaceVaseZoomed':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_zoomed.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'

        elif image_name == 'FaceVaseZoomedInvertColors':
            image_original = Image.open(os.path.join(self.Imagedir_path, "face_vase_zoomedinvertcolors.png")).convert('RGB')
            # reverse the colors
            image_original = ImageOps.invert(image_original)
            cmap = 'gray'
                
        elif image_name == 'KanizsaSqRot':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Kanizsa_square_rotated.jpg")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'KanizsaSqSlightlyRot':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Kanizsa_square_slightly_rotated.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'KanizsaSqToofar':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Kanizsa_square_toofar.png")).convert('RGB')
            cmap = 'gray'
        
        elif image_name == 'KanizsaRealContours':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Kanizsa_square_realContours.jpg")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'ICwcfg1black':
            image_original = Image.open(os.path.join(self.Imagedir_path, "ICwcfg1black.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'ICwcfg1onlyrec':
            image_original = Image.open(os.path.join(self.Imagedir_path, "ICwcfg1_onlyrec.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'EagleBW':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'schematic_images', "volture_bw.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'EagleTree':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'schematic_images', "n01614925_n01614925_17660 copy.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'EagleTest':
            image_original = Image.open(os.path.join(self.Imagedir_path,  "eagle_test.jpg")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'CloudCat':    
            image_original = Image.open(os.path.join(self.Imagedir_path, "CloudCat2.jpg")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'CloudDog':    
            image_original = Image.open(os.path.join(self.Imagedir_path, "clouddog1.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'CatElephantSilo':
            image_original = Image.open(os.path.join(self.Imagedir_path, "cat-elephant-silo.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'CatEdges':
            image_original = Image.open(os.path.join(self.Imagedir_path, "cat-edges.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'BearBottle':
            image_original = Image.open(os.path.join(self.Imagedir_path, "bear-bottle.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'Plug':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Danish_electrical_plugs.jpg")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'ImagenetHardFish':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_hard', "image.jpg")).convert('RGB')
            cmap = 'gray'   
        elif image_name == 'ImageNetTestSample':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_test', "ILSVRC2012_val_00025091.JPEG")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'VGGfaceTestSample':    
            image_original = Image.open(os.path.join(self.Imagedir_path, 'vggface_test', "vggface_sample.jpg")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'PerceptualBorder':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Paolo_natimage002.2400x2400.jpeg")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'Rorschach05':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'Rorschach_images',"Rorschach_blot_05.jpg")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'TextureShapeBearBottle':
            image_original = Image.open(os.path.join(self.Imagedir_path, "bear-bottle.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'TextureShapeCat5Knife2':
            image_original = Image.open(os.path.join(self.Imagedir_path, "cat5-knife2.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'FigureGroundLamme':
            image_original = Image.open(os.path.join(self.Imagedir_path, "figure_ground.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'FigureGroundLammeBordersDetected':
            image_original = Image.open(os.path.join(self.Imagedir_path, "FigureGroundLamme_last_pattern.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'FigureGroundTexture':
            image_original = Image.open(os.path.join(self.Imagedir_path, "FigureGroundNew_Figure.png")).convert('RGB')
            cmap = 'gray'
        
        elif image_name == 'FigureGroundUniform':
            image_original = Image.open(os.path.join(self.Imagedir_path, "FigureGroundNew_Uniform.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'FigureGroundCircle':
            image_original = Image.open(os.path.join(self.Imagedir_path, "FigureGroundCircle.png")).convert('RGB')
            cmap = 'gray'
        
        
        elif image_name == 'BorderOwenerShip1':
            image_original = Image.open(os.path.join(self.Imagedir_path, "border_ownership1.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'BorderOwenerShip2':
            image_original = Image.open(os.path.join(self.Imagedir_path, "border_ownership2.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'NeonCircles':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Neon_Color_Circle.jpg")).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'NeonColorSaeedi':
            image_original = Image.open(os.path.join(self.Imagedir_path, "NeonColorSaeedi.jpg")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'NeonCirclesCocenter':
            image_original = Image.open(os.path.join(self.Imagedir_path, "Neon_colour_spreading_illusion_no_caption.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'CrossGabor':
            image_original = Image.open(os.path.join(self.Illusiondir_path, "cross_gabor.jpg")).convert('RGB')
            cmap = 'gray'
        
        elif image_name == 'EgyptianMau13':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_test', "Egyptian Mau_13.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'EgyptianMau25':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_test', "Egyptian Mau_25.png")).convert('RGB')
            cmap = 'gray'
        elif image_name == 'EgyptianMauTrain25':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_train', "Egyptian Mau_train25.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'EgyptianMauTrain1':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'imagenet_train', "Egyptian Mau_train1.png")).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'RandomizedPhaseoval':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'RandomizedPhaseOval.png')).convert('RGB')
            cmap = 'Spectral'
        
        elif image_name == 'RandomizedPhaseovalGray':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'RandomizedPhaseOvalGray.png')).convert('RGB')
            cmap = 'Spectral'
            
        elif image_name == 'ConnectedDots':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ConnectedDots.png')).convert('RGB')
            cmap = 'gray'
        elif image_name == 'UnconnectedDots':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'UnconnectedDots.png')).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'Panda':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'panda.png')).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'GoodGestaltA':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GoodGestaltA.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'GoodGestaltB':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GoodGestaltB.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'GoodGestaltC':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GoodGestaltC.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'GoodGestaltD':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GoodGestaltD.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ContinuityGestaltA':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ContinuityGestaltA.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ContinuityGestaltB':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ContinuityGestaltB.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ContinuityGestaltZoomOutA':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ContinuityGestaltZoomOutA.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ContinuityGestaltZoomOutB':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ContinuityGestaltZoomOutB.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ContinuityGestaltZoomOutD':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ContinuityGestaltZoomOutD.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'GroupingByColor':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GroupingByColorSimilarity.png')).convert('RGB')
            cmap = 'Spectral'
            
        elif image_name == 'GroupingByContinuity':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'GroupingByContinuity.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ShadowEffect':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ShadowEffect.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'ShadowEffectBW':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'ShadowEffectBW.png')).convert('RGB')
            cmap = 'gray'
        elif image_name == 'ConfettiIllusion':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'Confetti_illusion.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'CornsweetBlock':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'CornsweetBlock.png')).convert('RGB')
            cmap = 'Spectral'
            
        elif image_name == 'EhrensteinSquare':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'EhrensteinSquare.png')).convert('RGB')
            cmap = 'gray'
            
        elif image_name == 'EhrensteinColor':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'EhrensteinColor.png')).convert('RGB')
            cmap = 'Spectral'
            
        elif image_name == 'EhresteinSingleColor':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'EhresteinSingleColor.png')).convert('RGB')
            cmap = 'Spectral'
        elif image_name == 'NoisyCat':
            image_original = Image.open(os.path.join(self.Imagedir_path, 'NoisyCat.png')).convert('RGB')
            cmap = 'Spectral'
        else:

            raise ValueError(f"Unsupported image name: {image_name}")

        return image_original, cmap