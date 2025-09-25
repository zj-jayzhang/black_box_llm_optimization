from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
import numpy as np
from PIL import Image
import cv2
from vlm_attacker.data import SampledImageDataset
from vlm_attacker.base_attacker import BaseAttacker

console = Console()




class QwenAttacker(BaseAttacker):
    """
    Attack framework for Qwen2.5-VL models.
    
    This class provides adversarial attack methods specifically for Qwen2.5-VL models,
    including white-box attacks that leverage model internals.
    """
    
    def __init__(self, model_name="3b", use_flash_attention=False, min_pixels=None, max_pixels=None, num_samples=100, targeted_attack=False, args=None):
        """
        Initialize the AttackQwen class with a Qwen2.5-VL model.
        
        Args:
            model_name (str): Model size ("3b", "7b", "72b")
            use_flash_attention (bool): Whether to use flash attention
            min_pixels (int): Minimum pixels for processor
            max_pixels (int): Maximum pixels for processor
        """

        self.use_flash_attention = use_flash_attention
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        
        super().__init__(model_name, num_samples, targeted_attack, args)
        self.device = next(self.model.parameters()).device
        
    def _initialize_model(self):
        """Initialize the Qwen2.5-VL model and processor."""
        if self.use_flash_attention:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype="auto", 
                device_map="auto"
            )

        processor_kwargs = {}
        if self.min_pixels is not None and self.max_pixels is not None:
            processor_kwargs.update({
                'min_pixels': self.min_pixels,
                'max_pixels': self.max_pixels
            })
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs
        )
    
    def _get_supported_attack_types(self):
        """Return list of supported attack types for Qwen models."""
        return ["black_box_attack_w_logprob", "black_box_attack_wo_logprob", "white_box_attack"]
    
    def get_logprobs_from_vlm(self, x, pixel_values=None, image_grid_thw=None, prompt=""):
        """
        Get logprobs from Qwen VLM for an image.
        
        Args:
            x: Input image (numpy array in [0,1] or PIL Image)
            pixel_values: Optional preprocessed pixel values
            image_grid_thw: Optional image grid tensor
            prompt: Text prompt to use
            
        Returns:
            tuple: (logprob, text_response)
        """
        if isinstance(x, np.ndarray):
            x = Image.fromarray((x * 255).astype(np.uint8))
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": x},
            {"type": "text", "text": prompt}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_normalize=False,
            do_resize=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if pixel_values is not None and image_grid_thw is not None:
                inputs["pixel_values"] = pixel_values
                inputs["image_grid_thw"] = image_grid_thw

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[0, -1]
            logprobs = F.log_softmax(last_token_logits, dim=-1)
            yes_token = self.processor.tokenizer.encode("1", add_special_tokens=False)[0]  
            logprob = logprobs[yes_token].item()
    
        try:
            response = generated_text.split('assistant')[-1].strip()
            if 'system' in response:
                response = response.split('system')[0].strip()
            if 'user' in response:
                response = response.split('user')[0].strip()
        except:
            response = generated_text
        
        return logprob, response

    def reconstruct_images_from_patches(self, flatten_patches, grid_shape, 
                                      temporal_patch_size=2, patch_size=14, 
                                      merge_size=2, channel=3, plot=False):
        """
        Reconstruct images from flattened patches.
        
        Args:
            flatten_patches: Flattened patches array
            grid_shape: Tuple of (grid_t, grid_h, grid_w) from preprocessing
            temporal_patch_size: Temporal patch size used in preprocessing
            patch_size: Spatial patch size used in preprocessing  
            merge_size: Merge size used in preprocessing
            channel: Number of channels in the image
            plot: Whether to save the reconstructed image
        
        Returns:
            Reconstructed image tensor
        """
        grid_t, grid_h, grid_w = grid_shape
        image_h, image_w = grid_h * patch_size, grid_w * patch_size
        
        # Reshape and reconstruct
        patches_b = flatten_patches.reshape(1, grid_h // merge_size, grid_w // merge_size, 
                                          merge_size, merge_size, channel, 
                                          temporal_patch_size, patch_size, patch_size)
        patches = patches_b.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        patches = patches.reshape(1, 2, 3, 224, 224)
        patches_img = patches.squeeze(0)

        if plot:
            image = patches_img[0]  
            assert image.shape == (channel, image_h, image_w)
            image_np = image.detach().cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            image_pil = Image.fromarray(image_np)
            image_pil.save('saved_results/reconstructed.png')
            
        return patches_img[0]

    def white_box_attack(self, source_x, x_original, verbose=False, iterations=1000, eps=16, prompt=""):
        """
        White-box attack using gradients with L∞ constraint.
        This attack has access to the model's internal representations.
        
        Args:
            source_x: Source image (numpy array in [0,1])
            iterations: Number of optimization iterations
            eps: Maximum perturbation magnitude (L∞ norm)
            prompt: Text prompt to use
            
        Returns:
            tuple: (adversarial_pixel_values, attack_success)
        """
        alpha = 0.05
        attack_success = False
        eps = eps / 255.0
        
        
        def make_inputs(img_np):
            pil = Image.fromarray((img_np * 255).astype("uint8"))
            messages = [{
                "role":"user",
                "content":[
                    {"type":"image","image": pil},
                    {"type":"text","text":prompt}
                ]
            }]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[chat],
                images=img_inputs,
                videos=vid_inputs,
                padding=True,
                return_tensors="pt",
                do_normalize=False,
                do_resize=False
            )
            return {k:v.to(self.device) for k,v in inputs.items()}

        src_inputs = make_inputs(source_x)
        x_orig = src_inputs["pixel_values"].detach().clone()
        x_adv = x_orig.clone()
        x_adv.requires_grad = True
        grid_thw = src_inputs["image_grid_thw"]
        
        best_loss = float("inf") if not self.targeted_attack else float("-inf")
        best_x = x_adv.clone().detach()

        for i in range(iterations):
            if x_adv.grad is not None:
                x_adv.grad.zero_()
                
            outputs = self.model(**{
                "pixel_values": x_adv,
                "image_grid_thw": grid_thw,
                **{k:v for k,v in src_inputs.items() if k not in ["pixel_values", "image_grid_thw"]}
            })
            
            last_token_logits = outputs.logits[0, -1]
            logprobs = F.log_softmax(last_token_logits, dim=-1)
            yes_token = self.processor.tokenizer.encode("1", add_special_tokens=False)[0]
            logprob = logprobs[yes_token]
            
            loss = logprob  # Minimize logprob of correct class

            loss.backward()
            grad = x_adv.grad.sign()
            
            with torch.no_grad():
                x_adv = x_adv - alpha * grad
                delta = x_adv - x_orig
                delta = torch.clamp(delta, -eps, eps)
                x_adv = x_orig + delta
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv.requires_grad = True
                
            if (loss.item() < best_loss and not self.targeted_attack) or (loss.item() > best_loss and self.targeted_attack):
                best_loss = loss.item()
                best_x = x_adv.clone().detach()
                
                eps_curr = np.abs(best_x.detach().cpu().numpy() - x_orig.detach().cpu().numpy()).max()
                
                new_img = self.reconstruct_images_from_patches(
                    flatten_patches=best_x, grid_shape=grid_thw[0].tolist(),
                )
                new_img = np.transpose(new_img.detach().cpu().numpy(), (1, 2, 0))
                curr_logprob, curr_text = self.get_logprobs_from_vlm(new_img, prompt=prompt)
                if verbose:
                    console.print(Panel(f"[cyan] Loss: {loss.item():.6f}, Current logprob: {curr_logprob:.4f}, eps: {255*eps_curr}, \n response: {curr_text}[/cyan]", 
                                        title="White-box Attack"))
                if (curr_text.startswith("0") and not self.targeted_attack) or (curr_text.startswith("1") and self.targeted_attack):
                    attack_success = True
                    break
                
        return best_x, attack_success

    def _get_comparison_response(self, img_1, img_2, prompt):
        """
        Qwen-specific method to get comparative response between two images.
        
        Args:
            img_1: First image (CHW numpy array, uint8)
            img_2: Second image (CHW numpy array, uint8)
            prompt: Comparison prompt
            
        Returns:
            response_text
        """
        # Build Qwen message format with two images
        pil1 = Image.fromarray(np.transpose(img_1, (1, 2, 0)).astype(np.uint8))
        pil2 = Image.fromarray(np.transpose(img_2, (1, 2, 0)).astype(np.uint8))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil1},
                {"type": "image", "image": pil2},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process with Qwen-specific pipeline
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_normalize=False,
            do_resize=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
        response = generated_text.strip().split("assistant")[-1].strip()
        return response 


