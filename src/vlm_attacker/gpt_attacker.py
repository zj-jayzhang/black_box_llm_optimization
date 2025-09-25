from openai import OpenAI
import torch
from rich.console import Console
from rich.panel import Panel
import numpy as np
from PIL import Image
import base64
import os
import uuid
from vlm_attacker.data import SampledImageDataset
from vlm_attacker.base_attacker import BaseAttacker
import dotenv

dotenv.load_dotenv()

console = Console()




class GPTAttacker(BaseAttacker):
    """
    Attack framework for GPT vision models.
    
    This class provides adversarial attack methods specifically for OpenAI's GPT models
    with vision capabilities.
    """
    
    def __init__(self, model_name="gpt-4o-mini", num_samples=100, targeted_attack=False, args=None):
        """
        Initialize the AttackGPT class with an OpenAI GPT model.
        
        Args:
            model_name (str): Model name ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o")
        """
        
        super().__init__(model_name, num_samples, targeted_attack, args)
        
    def _initialize_model(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _get_supported_attack_types(self):
        """Return list of supported attack types for GPT models."""
        return ["black_box_attack_w_logprob", "black_box_attack_wo_logprob"]
    
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
    
    def get_logprobs_from_vlm(self, x, prompt="", best_logprob=100):
        """
        Get logprobs from GPT model for an image.
        
        Args:
            x: Input image (numpy array in [0,1] or PIL Image)
            prompt: the prompt to use
            best_logprob: fallback logprob value
            
        Returns:
            tuple: (logprob, text_response)
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

            # Getting the Base64 string
            base64_image = self._encode_image(temp_path)
            
            # Delete temp file immediately after encoding - we don't need it anymore
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if "gpt-5-mini" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "low"
                                    },
                                },
                            ],
                        }
                    ],
                    #! max_tokens and logprobs are not supported for gpt-5-mini
                    # max_tokens=300,
                    # logprobs=True,
                    # top_logprobs=20,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "low"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    logprobs=True,
                    top_logprobs=20,
                )
            if response.choices[0].logprobs:
                next_token_probs = response.choices[0].logprobs.content[0].top_logprobs
                next_token_list = [token.token for token in next_token_probs]
                next_token_logprobs = [token.logprob for token in next_token_probs]
                #todo: maybe it's easier to ask the model in a way that it will output 1 more often, especially for targeted attack
                yes_prob = next_token_logprobs[next_token_list.index("1")] if "1" in next_token_list else best_logprob
            else:
                yes_prob = best_logprob
            # top1_token_prob = response.choices[0].logprobs.content[0].top_logprobs
            # top_token = top1_token_prob[0].token
            # top_logprob = top1_token_prob[0].logprob
            # yes_prob = top_logprob if top_token == "1" else best_logprob - 0.01


            text_response = response.choices[0].message.content
            return yes_prob, text_response
            
        except Exception as e:
            # Clean up temp file in case of error before encoding
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def _get_comparison_response(self, img_1, img_2, prompt):
        """
        GPT-specific method to get comparative response between two images.
        
        Args:
            img_1: First image (CHW numpy array, uint8)
            img_2: Second image (CHW numpy array, uint8)
            prompt: Comparison prompt
            
        Returns:
            tuple: (response_text, logprob_1, logprob_0)
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
            
            # Delete temp files immediately after encoding - we don't need them anymore
            for temp_path in [temp_path1, temp_path2]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Build OpenAI API message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img1}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img2}",
                                "detail": "low"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Retry up to 5 times
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        #! max_tokens and logprobs are not supported for gpt-5-mini
                        # max_tokens=300,
                        # logprobs=True,
                        # top_logprobs=1,
                    )
                                
                    text_response = response.choices[0].message.content
                    return text_response
                    
                except Exception as e:
                    console.print(f"[yellow]Attempt {attempt + 1}/5 failed in _get_comparison_response: {str(e)}[/yellow]")
                    if attempt == 4:  # Last attempt
                        console.print(f"[red]All 5 attempts failed in _get_comparison_response: {str(e)}[/red]")
                        return str(e)
                    # Wait a bit before retrying (optional)
                    import time
                    time.sleep(1)
                    
        except Exception as e:
            # Clean up temp files in case of error before encoding
            for temp_path in [temp_path1, temp_path2]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise e

