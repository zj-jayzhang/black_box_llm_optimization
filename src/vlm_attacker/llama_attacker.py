from transformers import MllamaForConditionalGeneration, AutoProcessor
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


class LlamaAttacker(BaseAttacker):
    """
    Attack framework for Llama Vision models.
    
    This class provides adversarial attack methods specifically for Llama Vision models,
    including white-box attacks that leverage model internals.
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", use_flash_attention=False, num_samples=100, targeted_attack=False, args=None):
        """
        Initialize the LlamaAttacker class with a Llama Vision model.
        
        Args:
            model_name (str): Model name/path (e.g., "meta-llama/Llama-3.2-11B-Vision-Instruct")
            use_flash_attention (bool): Whether to use flash attention
        """
        self.use_flash_attention = use_flash_attention
        self.model_name = model_name
        
        super().__init__(model_name, num_samples, targeted_attack, args)
        self.device = next(self.model.parameters()).device
        
    def _initialize_model(self):
        """Initialize the Llama Vision model and processor."""
        if self.use_flash_attention:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def _get_supported_attack_types(self):
        """Return list of supported attack types for Llama models."""
        return ["black_box_attack_w_logprob", "black_box_attack_wo_logprob", "white_box_attack"]
    
    def get_logprobs_from_vlm(self, x, prompt="", **kwargs):
        """
        Get logprobs from Llama VLM for an image.
        
        Args:
            x: Input image (numpy array in [0,1] or PIL Image)
            prompt: Text prompt to use
            **kwargs: Additional model-specific parameters
            
        Returns:
            tuple: (logprob, text_response)
        """
        if isinstance(x, np.ndarray):
            x = Image.fromarray((x * 255).astype(np.uint8))
        
        #! save the image for debugging
        x.save("debug_image.png")
        # Use proper Llama Vision message format
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Apply chat template and process
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            x,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Get logprobs for the "1" token
            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[0, -1]
            logprobs = F.log_softmax(last_token_logits, dim=-1)
            yes_token = self.processor.tokenizer.encode("1", add_special_tokens=False)[0]
            logprob = logprobs[yes_token].item()
        
        # Extract response text after the input
        try:
            # Extract everything after the "assistant" marker
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text.strip()
        except:
            response = generated_text
        return logprob, response

    def white_box_attack(self, source_x, iterations=1000, eps=16, prompt=""):
        """
        White-box attack using gradients with L∞ constraint.
        This attack has access to the model's internal representations.
        
        Args:
            source_x: Source image (numpy array in [0,1])
            iterations: Number of optimization iterations
            eps: Maximum perturbation magnitude (L∞ norm)
            prompt: Text prompt to use
            
        Returns:
            tuple: (adversarial_image, attack_success)
        """
        #! TODO: Implement white-box attack for Llama Vision models
        # check https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py#L601
        #  inputs['pixel_values']: torch.Size([1, 1, 4, 3, 560, 560])
        #  batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width
        pass

    def _get_comparison_response(self, img_1, img_2, prompt):
        """
        Llama-specific method to get comparative response between two images.
        
        Args:
            img_1: First image (CHW numpy array, uint8)
            img_2: Second image (CHW numpy array, uint8) 
            prompt: Comparison prompt
            
        Returns:
            response_text
        """
        # Convert images to proper format
        pil1 = Image.fromarray(np.transpose(img_1, (1, 2, 0)).astype(np.uint8))
        pil2 = Image.fromarray(np.transpose(img_2, (1, 2, 0)).astype(np.uint8))
        
        # Build Llama message format with two images (similar to Qwen)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"}, 
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Process with Llama-specific pipeline
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            [pil1, pil2],  # Pass both images as a list
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        # Extract response text after the input
        try:
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text.strip()
        except:
            response = generated_text
            
        return response  # Single response from comparing both images
