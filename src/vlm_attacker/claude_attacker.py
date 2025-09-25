import anthropic
import torch
from rich.console import Console
from rich.panel import Panel
import numpy as np
from PIL import Image
import base64
import os
import uuid
import tempfile
from vlm_attacker.data import SampledImageDataset
from vlm_attacker.base_attacker import BaseAttacker
import dotenv

dotenv.load_dotenv()

console = Console()


class ClaudeAttacker(BaseAttacker):
    """
    Attack framework for Claude vision models.
    
    This class provides adversarial attack methods specifically for Anthropic's Claude models
    with vision capabilities.
    """
    
    def __init__(self, model_name="claude-3-5-sonnet-20241022", num_samples=100, targeted_attack=False, args=None):
        """
        Initialize the ClaudeAttacker class with an Anthropic Claude model.
        
        Args:
            model_name (str): Model name ("claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", etc.)
            num_samples (int): Number of samples for attack evaluation
        """
        
        super().__init__(model_name, num_samples, targeted_attack, args)
        
    def _initialize_model(self):
        """Initialize the Anthropic client."""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def _get_supported_attack_types(self):
        """Return list of supported attack types for Claude models."""
        # Claude doesn't provide logprobs, so we only support the logprob-free attack
        return ["black_box_attack_wo_logprob"]
    
    def _encode_image(self, image_path):
        """
        Function to encode the image to base64.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_media_type(self, image_path):
        """
        Get media type from image path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Media type for the image
        """
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return "image/jpeg"
        elif ext == '.png':
            return "image/png"
        elif ext == '.gif':
            return "image/gif"
        elif ext in ['.webp']:
            return "image/webp"
        else:
            # Default to PNG for unknown extensions
            return "image/png"
    
    def get_logprobs_from_vlm(self, x, prompt="", best_logprob=100):
        """
        Get response from Claude model for an image.
        
        Note: Claude doesn't provide logprobs, so we return a default value.
        This method is kept for interface compatibility.
        
        Args:
            x: Input image (numpy array in [0,1] or PIL Image)
            prompt: the prompt to use
            best_logprob: fallback logprob value (returned as default)
            
        Returns:
            tuple: (default_logprob, text_response)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(x, np.ndarray):
            x = Image.fromarray((x * 255).astype(np.uint8))
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Create unique temporary file name to avoid race conditions
        unique_id = str(uuid.uuid4())
        temp_path = f"temp/temp_image_{unique_id}.png"
        
        try:
            x.save(temp_path)

            # Getting the Base64 string and media type
            base64_image = self._encode_image(temp_path)
            media_type = self._get_media_type(temp_path)
            
            # Delete temp file immediately after encoding - we don't need it anymore
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            text_response = message.content[0].text
            
            # Claude doesn't provide logprobs, so we return a default value
            # For binary classification, we can try to infer from the response
            return 0.0, text_response  
                
        except Exception as e:
            # Clean up temp file in case of error before encoding
            if os.path.exists(temp_path):
                os.remove(temp_path)
            console.print(f"[red]Error in get_logprobs_from_vlm: {str(e)}[/red]")
            return best_logprob - 0.01, str(e)
    
        
    def _get_comparison_response(self, img_1, img_2, prompt):
        """
        Claude-specific method to get comparative response between two images.
        
        Args:
            img_1: First image (CHW numpy array, uint8)
            img_2: Second image (CHW numpy array, uint8)
            prompt: Comparison prompt
            
        Returns:
            str: Response text from Claude
        """
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Create unique temporary file names to avoid race conditions
        unique_id = str(uuid.uuid4())
        temp_path1 = f"temp/temp_img1_{unique_id}.png"
        temp_path2 = f"temp/temp_img2_{unique_id}.png"
        
        try:
            pil1 = Image.fromarray(np.transpose(img_1, (1, 2, 0)).astype(np.uint8))
            pil2 = Image.fromarray(np.transpose(img_2, (1, 2, 0)).astype(np.uint8))
            
            pil1.save(temp_path1)
            pil2.save(temp_path2)
            
            base64_img1 = self._encode_image(temp_path1)
            base64_img2 = self._encode_image(temp_path2)
            
            media_type1 = self._get_media_type(temp_path1)
            media_type2 = self._get_media_type(temp_path2)
            
            # Delete temp files immediately after encoding - we don't need them anymore
            for temp_path in [temp_path1, temp_path2]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Build Anthropic API message format
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type1,
                        "data": base64_img1,
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type2,
                        "data": base64_img2,
                    },
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # Retry up to 5 times
            for attempt in range(5):
                try:
                    message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=300,
                        messages=[
                            {
                                "role": "user",
                                "content": content
                            }
                        ],
                    )
                                
                    text_response = message.content[0].text
                    return text_response
                    
                except Exception as e:
                    console.print(f"[yellow]Attempt {attempt + 1}/5 failed in _get_comparison_response: {str(e)}[/yellow]")
                    if attempt == 4:  # Last attempt
                        console.print(f"[red]All 5 attempts failed in _get_comparison_response: {str(e)}[/red]")
                        return str(e)
                    # Wait a bit before retrying
                    import time
                    time.sleep(1)
                    
        except Exception as e:
            # Clean up temp files in case of error before encoding
            for temp_path in [temp_path1, temp_path2]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise e 