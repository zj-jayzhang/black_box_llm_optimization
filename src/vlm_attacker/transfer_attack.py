import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torchvision
# OpenAI GPT-4o integration
import base64
import os
import sys
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vlm_attacker.data import SampledImageDataset

#! This code is mainly from https://github.com/VILA-Lab/M-Attack, thanks to the authors for the code.
from encoder import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_image(image_path):
    """Encode an image file to base64 string for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image(image, prompt="Describe this image in detail.", model_name="gpt-4o"):
    """
    Use GPT-4o to describe an image.
    
    Args:
        image: Input image (numpy array in [0,1], PIL Image, or torch.Tensor)
        prompt: Text prompt for description
        model_name: GPT model to use
        
    Returns:
        str: GPT-4o's description of the image
    """
    # Convert input to PIL Image
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy array
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        else:  # HWC format
            image_np = image.detach().cpu().numpy()
        # Ensure values are in [0, 1] range and convert to uint8
        image_np = np.clip(image_np, 0, 1)
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    elif isinstance(image, np.ndarray):
        # Handle numpy array
        if image.ndim == 3 and image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC
        # Ensure values are in correct range
        if image.max() <= 1.0:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure RGB mode
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Save image temporarily
    temp_path = "temp_describe_image.png"
    pil_image.save(temp_path)
    
    try:
        # Convert image to base64
        base64_image = encode_image(temp_path)
        
        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        
        description = response.choices[0].message.content
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return description
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"Error calling GPT-4o: {str(e)}"


# Transform PIL.Image to PyTorch Tensor (matching the original)
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

class Transferable_adv_data:
    """
    Minimal working example for generating transferable adversarial samples using CLIP models.
    Uses FGSM attack with source cropping enabled.
    """
    
    def __init__(self, 
                 device: str = 'cuda:0',  # Match config
                 input_res: int = 224,
                 epsilon: float = 16.0,   # Keep in 0-255 range
                 alpha: float = 1.0,      # Match config: 1.0 instead of 2.0
                 steps: int = 300,        # Match config
                 crop_scale: tuple = (0.5, 0.9)):  # Match config: [0.5, 0.9]
        """
        Initialize the transferable adversarial data generator.
        
        Args:
            device: Device to run computations on
            input_res: Input image resolution
            epsilon: Maximum perturbation budget (in pixel values 0-255)
            alpha: Step size for FGSM (in pixel values 0-255)
            steps: Number of optimization steps
            crop_scale: Scale range for random cropping
        """
        self.device = device
        self.input_res = input_res
        self.epsilon = epsilon  # Keep in 0-255 range
        self.alpha = alpha      # Keep in 0-255 range
        self.steps = steps
        self.crop_scale = crop_scale
        
        # Initialize CLIP models (only 3 models to match config)
        self.models = self._load_clip_models()
        self.ensemble_extractor = EnsembleFeatureExtractor(self.models)
        self.ensemble_loss = EnsembleFeatureLoss(self.models)
        
        # Source cropping transform (enabled by default)
        self.source_crop = transforms.RandomResizedCrop(
            self.input_res, 
            scale=self.crop_scale
        )
        
        # Target cropping transform  
        self.target_crop = transforms.RandomResizedCrop(
            self.input_res, 
            scale=self.crop_scale
        )
        
    def _load_clip_models(self) -> List[nn.Module]:
        """Load and initialize ensemble of CLIP models (only 3 models to match config)."""
        model_classes = [
            ClipB16FeatureExtractor,   # B16
            ClipB32FeatureExtractor,   # B32  
            ClipLaionFeatureExtractor  # Laion
            # Removed ClipL336FeatureExtractor to match config
        ]
        
        models = []
        for model_class in model_classes:
            model = model_class().eval().to(self.device).requires_grad_(False)
            models.append(model)
            
        return models
    
    def generate_adversarial_sample(self, 
                                  source_image: torch.Tensor,
                                  target_image: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial sample using FGSM with source cropping.
        
        Args:
            source_image: Source image tensor [C, H, W] in range [0, 255]
            target_image: Target image tensor [C, H, W] in range [0, 255]
            
        Returns:
            Adversarial image tensor [C, H, W] in range [0, 255]
        """
        # Ensure images are on correct device and have batch dimension
        if source_image.dim() == 3:
            source_image = source_image.unsqueeze(0)
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
            
        source_image = source_image.to(self.device)
        target_image = target_image.to(self.device)
        
        # Initialize perturbation
        delta = torch.zeros_like(source_image, requires_grad=True)
        
        # Progress bar for optimization
        pbar = tqdm(range(self.steps), desc="FGSM Attack Progress")
        
        # Main FGSM optimization loop
        for step in pbar:
            # Set target features (cropped target)
            with torch.no_grad():
                target_cropped = self.target_crop(target_image)
                self.ensemble_loss.set_ground_truth(target_cropped)
            
            # Forward pass with current adversarial image
            adv_image = source_image + delta
            
            # Get features from cropped adversarial image (source cropping enabled)
            local_cropped = self.source_crop(adv_image)
            local_features = self.ensemble_extractor(local_cropped)
            
            # Calculate similarity loss (we want to maximize similarity)
            local_sim = self.ensemble_loss(local_features)
            loss = local_sim
            
            # Calculate metrics for monitoring
            metrics = {
                "max_delta": torch.max(torch.abs(delta)).item(),
                "mean_delta": torch.mean(torch.abs(delta)).item(),
                "local_similarity": local_sim.item()
            }
            
            # Update progress bar
            pbar_metrics = {k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" 
                           for k, v in metrics.items()}
            pbar.set_postfix(pbar_metrics)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            
            # FGSM update (working with 0-255 range)
            delta.data = torch.clamp(
                delta + self.alpha * torch.sign(grad),
                min=-self.epsilon,
                max=self.epsilon
            )
        
        # Generate final adversarial image
        adv_image = source_image + delta
        adv_image = torch.clamp(adv_image, 0.0, 255.0)  # Clamp to 0-255 range
        
        # Remove batch dimension if added
        if adv_image.size(0) == 1:
            adv_image = adv_image.squeeze(0)
            
        # Convert to [0,1] range for saving
        adv_image_normalized = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        # torchvision.utils.save_image(adv_image_normalized, "adversarial_image.png")
        
        return adv_image  # Return in 0-255 range to match original
    
    def generate_batch(self, 
                      source_images: torch.Tensor,
                      target_images: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial samples for a batch of images.
        
        Args:
            source_images: Batch of source images [B, C, H, W]
            target_images: Batch of target images [B, C, H, W]
            
        Returns:
            Batch of adversarial images [B, C, H, W]
        """
        batch_size = source_images.size(0)
        adv_images = []
        
        for i in range(batch_size):
            adv_img = self.generate_adversarial_sample(
                source_images[i], 
                target_images[i]
            )
            adv_images.append(adv_img)
            
        return torch.stack(adv_images)


# Example usage
def main():
    """
    Example of how to use the Transferable_adv_data class.
    Generate adversarial images for all samples in the dataset.
    """
    import random
    import os
    
    # Initialize the adversarial generator with config-matching parameters
    adv_generator = Transferable_adv_data(
        device='cuda:0' if torch.cuda.is_available() else 'cpu',  # Match config
        input_res=224,
        epsilon=32.0,    # Max perturbation in [0,255] range
        alpha=1.0,       # Step size in [0,255] range (match config)
        steps=300,       # Number of FGSM iterations (match config)
        crop_scale=(0.5, 0.9)  # Random crop scale range (match config)
    )
    
    # Load all samples from dataset
    dataset = SampledImageDataset('data/imgnet_subset/', num_samples=None)
    print(f"Loaded dataset with {len(dataset)} images")
    
    # Create output directory
    img_saved_type = "target" # "target" or "untarget"
    output_dir = "data/adv_images_target" if img_saved_type == "target" else "data/adv_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Adversarial images will be saved to: {output_dir}/")
    
    def load_and_process_image(img_path):
        """Load and process a single image to tensor format [C, H, W] in range [0, 255]"""
        # Create transform pipeline matching the original
        transform_fn = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),  # Use custom to_tensor function
        ])
        
        # Load and transform image
        image = Image.open(img_path)
        x_tensor = transform_fn(image)  # Returns tensor in 0-255 range
        
        return x_tensor
    
    def get_random_target(source_idx, source_label, dataset):
        """Get a random target image with different label than source"""
        available_targets = []
        for i, (_, target_label, _) in enumerate(dataset):
            if i != source_idx and target_label != source_label:
                available_targets.append(i)
        
        if not available_targets:
            # Fallback: if no different label found, use a different image with same label
            available_targets = [i for i in range(len(dataset)) if i != source_idx]
        
        return random.choice(available_targets)
    
    # Set random seed for reproducible target selection
    random.seed(42)
    
    print(f"\n=== Generating Adversarial Images for All Samples ===")
    


    # Process each sample in the dataset
    for idx in range(len(dataset)):
        source_path, source_label, _ = dataset[idx]
        
        # Get random target with different label
        target_idx = get_random_target(idx, source_label, dataset)
        target_path, target_label, _ = dataset[target_idx]
        
        # Get filename early for error handling
        source_filename = os.path.basename(source_path)
        
        print(f"\n[{idx+1}/{len(dataset)}] Processing: {source_filename}")
        print(f"  Source: {source_label} -> Target: {target_label}")
        try:
            # Load and process images
            source_tensor = load_and_process_image(source_path)
            target_tensor = load_and_process_image(target_path)
            
            # Generate adversarial sample
            print(f"  Generating adversarial sample...")
            adv_image = adv_generator.generate_adversarial_sample(source_tensor, target_tensor)
            
            # Ensure both tensors are on the same device for comparison
            source_device = source_tensor.device
            adv_image_same_device = adv_image.to(source_device)
            source_tensor_same_device = source_tensor.to(source_device)
            
            # Calculate perturbation
            perturbation = torch.max(torch.abs(adv_image_same_device - source_tensor_same_device)).item()
            print(f"  Max perturbation: {perturbation:.1f} (out of 255)")
            
            # Save adversarial image with same suffix as source
            source_name, source_ext = os.path.splitext(source_filename)
            #!Note: this is actually the same, but the experiments are too expensive to run again, so we save the images again.
            if img_saved_type == "target":
                adv_filename = f"{source_name}_{target_label}{source_ext}"
            else:
                adv_filename = f"{source_name}_adv{source_ext}"

            adv_path = os.path.join(output_dir, adv_filename)
            
            # Convert to [0,1] range for saving
            adv_image_normalized = torch.clamp(adv_image / 255.0, 0.0, 1.0)
            torchvision.utils.save_image(adv_image_normalized, adv_path)
            print(f"  Saved: {adv_filename}")
            
            # Optional: Use GPT-4o to describe changes (for first few samples)
            if idx < 3:  # Only describe first 3 to save API calls
                print(f"  Getting GPT-4o descriptions...")
                original_desc = describe_image(source_tensor / 255.0, "What is the main object in this image?")
                adv_desc = describe_image(adv_image / 255.0, "What is the main object in this image?")
                
                print(f"  Original: {original_desc}")
                print(f"  Adversarial: {adv_desc}")
            
        except Exception as e:
            print(f"  Error processing {source_filename}: {str(e)}")
            continue
    
    print(f"\n=== Completed! ===")
    print(f"Generated adversarial images for {len(dataset)} samples")
    print(f"Check the '{output_dir}/' directory for results")


if __name__ == "__main__":
    main()