from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from rich.console import Console
from rich.panel import Panel
from vlm_attacker.data import SampledImageDataset

console = Console()


class BaseAttacker(ABC):
    """
    Abstract base class for adversarial attacks on vision-language models.
    
    This class provides a common interface and shared functionality for attacking
    different types of VLM models (Qwen, GPT, Claude, Llama, etc.).
    """
    
    def __init__(self, model_name, num_samples=100, targeted_attack=False, args=None):
        """
        Initialize the base attacker.
        
        Args:
            model_name (str): Name/identifier of the model to attack
        """
        self.model_name = model_name
        self.attack_success_rate = str(0) + "/" + str(num_samples)
        self.targeted_attack = targeted_attack
        self.gen_img = 0
        self.score_list = []
        self.sample_idx = 0
        self.args = args
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_logprobs_from_vlm(self, x, prompt="", **kwargs):
        """
        Get logprobs from the VLM for an image.
        
        Args:
            x: Input image (numpy array in [0,1] or PIL Image)
            prompt: Text prompt to use
            **kwargs: Additional model-specific parameters
            
        Returns:
            tuple: (logprob, text_response)
        """
        pass
    
    @abstractmethod
    def _get_supported_attack_types(self):
        """Return list of supported attack types for this model."""
        pass
    
    def _p_selection(self, p_init, it, n_iters):
        """
        Piece-wise constant schedule for p (the fraction of pixels changed on every iteration).
        
        Args:
            p_init (float): Initial fraction of pixels to perturb
            it (int): Current iteration number
            n_iters (int): Total number of iterations
            
        Returns:
            float: Current fraction of pixels to perturb
        """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p
    
    def _load_and_preprocess_image(self, image_path, img_size=224):
        """
        Load and preprocess image from path or URL.
        
        Args:
            image_path: Path to input image or URL
            img_size: Target image size
            
        Returns:
            numpy.ndarray: Preprocessed image in [0,1] range
        """
        # Load image
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy and resize
        x = np.array(image).astype(np.float32) / 255.0
        x = cv2.resize(x, (img_size, img_size))
        
        return x
    
    def _prepare_image_for_attack(self, source_x):
        """
        Prepare image for attack (convert format, ensure uint8, etc.).
        
        Args:
            source_x: Source image
            
        Returns:
            numpy.ndarray: Image ready for attack in CHW uint8 format
        """
        # Convert to CHW format
        if source_x.shape[0] != 3:
            source_x = np.transpose(source_x, (2, 0, 1))

        # Convert to uint8 if not already
        if source_x.dtype != np.uint8:
            source_x = (source_x * 255).astype(np.uint8) if source_x.max() <= 1.0 else source_x.astype(np.uint8)
        
        return source_x
    
    # for debug purpose, let's visualize the image
    def _save_comparison_plot(self, clean_img, transfer_img, ours_img, filename="comparison.png"):
        import matplotlib.pyplot as plt

        def to_display_format(img):
            img = np.asarray(img)
            
            # Convert from (3, H, W) to (H, W, 3) if needed
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            # If already (H, W, 3), no change
            elif img.ndim == 3 and img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}. Expected (3, H, W) or (H, W, 3)")

            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            return np.clip(img, 0, 1)


        images = [clean_img, ours_img]
        titles = ["x_adv", "x_new"]
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(to_display_format(img))
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_img(self, img, filename="comparison.png"):
        import matplotlib.pyplot as plt

        def to_display_format(img):
            img = np.asarray(img)
            
            # Convert from (3, H, W) to (H, W, 3) if needed
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            # If already (H, W, 3), no change
            elif img.ndim == 3 and img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}. Expected (3, H, W) or (H, W, 3)")

            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            return np.clip(img, 0, 1)


        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(to_display_format(img))
        ax.axis('off')
        ax.set_title(filename)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def black_box_attack_w_logprob(self, source_x, x_original, verbose=False, iterations=2000, prompt="", eps=8, **kwargs):
        """
        Black-box square attack using logprobs, with perturbation bounded w.r.t. original image.

        Args:
            source_x: Starting point (could be adversarial) to refine attack
            x_original: The true original image (perturbation measured w.r.t. this)
            iterations: Number of attack iterations
            prompt: Text prompt for the model
            eps: Maximum perturbation size (in pixel value)
            **kwargs: Additional attack parameters

        Returns:
            tuple: (best_adversarial_example, attack_success)
        """
        attack_success = False
        x_original = self._prepare_image_for_attack(x_original)
        x_curr = self._prepare_image_for_attack(source_x)
        x_curr_transposed = np.transpose(x_curr, (1, 2, 0))
        init_logprob, init_text = self.get_logprobs_from_vlm(
            x=x_curr_transposed.astype(np.float32) / 255.0, prompt=prompt
        )
        console.print(f"[yellow]Initial prob: {np.exp(init_logprob):.4f}, response: {init_text}[/yellow]")
        np.random.seed(0)
        min_val, max_val = 0, 255
        c, h, w = x_curr.shape
        n_features = c * h * w

        assert isinstance(eps, int), "eps must be an integer"

        init_delta = np.random.choice([-eps, eps], size=[c, 1, w])
        x_best = np.clip(x_curr + init_delta, min_val, max_val)

        # 🔐 Project x_best to be within eps of x_original
        delta_from_original = np.clip(x_best - x_original, -eps, eps)
        x_best = np.clip(x_original + delta_from_original, min_val, max_val)

        n_queries = 1
        n_queries_per_iter = 1  # kwargs.get('n_queries_per_iter', 5)
        best_logprob = np.inf if not self.targeted_attack else -np.inf

   
        # prompt_score = "Does this image contain a dog? Assign a confidence score from 1 to 100. 1 means very unlikely, 100 means very likely. Please output the confidence score first, then explain your reasoning. The confidence score should be a number between 1 and 100.Use the format: The confidence score is <confidence_score>. <explanation>"

        for i_iter in range(iterations):
            p = self._p_selection(0.5, i_iter, iterations)
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h - 1)

            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            delta = x_best - x_curr

            x_curr_window = x_curr[:, center_h:center_h + s, center_w:center_w + s]
            x_best_window = x_best[:, center_h:center_h + s, center_w:center_w + s]

            attempt = 0
            while np.sum(np.abs(np.clip(x_curr_window + delta[:, center_h:center_h + s, center_w:center_w + s], min_val, max_val) - x_best_window) < 1e-7) == c * s * s:
                delta[:, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps], size=[c, 1, 1])
                attempt += 1
                if attempt > 10:
                    break

            x_new = np.clip(x_curr + delta, min_val, max_val)
            
            # 🧩 Clip to keep within eps of x_original
            delta_from_original = np.clip(x_new - x_original, -eps, eps)
            x_new = np.clip(x_original + delta_from_original, min_val, max_val)

            # Optional assertion (can be commented out in production)
            assert np.abs((x_new - x_original).astype(np.int16)).max() <= eps

            # Query model
            x_new_transposed = np.transpose(x_new, (1, 2, 0))
            curr_logprob = 0
            for i in range(n_queries_per_iter):
                logprob, curr_text = self.get_logprobs_from_vlm(
                    x_new_transposed.astype(np.float32) / 255.0, prompt=prompt
                )
                curr_logprob += logprob

            # _, curr_text_score = self.get_logprobs_from_vlm(
            #     x_new_transposed.astype(np.float32) / 255.0, prompt=prompt_score
            # )
            # curr_score = int(curr_text_score.split("The confidence score is ")[1].split(".")[0])


            curr_logprob /= n_queries_per_iter
            n_queries += n_queries_per_iter
            #! we want to minimize the logprob of token '1' for non-targeted attack, and maximize the logprob of token '1' for targeted attack
            if (curr_logprob < best_logprob and not self.targeted_attack) or (curr_logprob > best_logprob and self.targeted_attack):
                best_logprob = curr_logprob
                x_best = x_new
                
                eps_curr = np.abs((x_best - x_original).astype(np.int16)).max()
                console.print(Panel(
                    f"[red]Iteration {i_iter}, best prob of 1={np.exp(best_logprob):.4f}, square size={s}, eps={eps_curr}, Queries={n_queries}, Text:\n{curr_text}[/red]",
                    title=f"Black-box Attack, success rate: {self.attack_success_rate}"
                ))
                # print(f"logprob: {curr_logprob}")
                # print(f"curr_prob: {np.exp(curr_logprob)}, curr_score: {curr_score}")
                # self.score_list.append([np.exp(curr_logprob), curr_score])
                # np.save(f"plot/setting_a1/{self.gen_img}.npy", np.array(self.score_list))


                if (curr_text.startswith("0") and not self.targeted_attack) or (curr_text.startswith("1") and self.targeted_attack):
                    # self._save_comparison_plot(x_original, x_curr, x_best, filename=f"temp/adv_imgs/adv_clean_{self.gen_img}.png")
                    # if np.exp(best_logprob) <0.1:
                    attack_success = True
                    break

            elif i_iter % 10 == 0:

                console.print(Panel(
                    f"[green]Iteration {i_iter}: Current prob of 1={np.exp(curr_logprob):.4f}, Best prob of 1={np.exp(best_logprob):.4f}, Queries={n_queries}, Text:\n{curr_text}[/green]",
                    title=f"Black-box Attack, success rate: {self.attack_success_rate}"
                ))
        self.gen_img += 1

        return x_best, attack_success

    #!  try for targeted attack, two stages
    def black_box_attack_wo_logprob_copy(self, source_x, x_original, prompt, test_prompt, verbose=False, iterations=2000, eps=8, **kwargs):
        """
        Generic black-box attack without logprobs using comparative responses.
        Supports both targeted and non-targeted attacks.
        
        Args:
            source_x: Starting point (could be adversarial) to refine attack
            x_original: The true original image (perturbation measured w.r.t. this)
            iterations: Number of attack iterations  
            prompt: Comparison prompt for the model
            test_prompt: Testing prompt to evaluate success
            eps: Maximum perturbation size in uint8 range [0,255]
            **kwargs: Additional model-specific parameters
            
        Returns:
            tuple: (best_adversarial_example, attack_success)
        """
        if not hasattr(self, '_get_comparison_response'):
            raise NotImplementedError("Subclass must implement _get_comparison_response method")
            
        attack_success = False   

        # Prepare images
        x_original = self._prepare_image_for_attack(x_original)
        source_x = self._prepare_image_for_attack(source_x)
        x_curr = source_x.copy()

        np.random.seed(0)
        min_val, max_val = 0, 255
        c, h, w = x_curr.shape
        n_features = c * h * w
        assert isinstance(eps, int), "eps must be an integer"

        init_delta = np.random.choice([-eps, eps], size=[c, 1, w])
        x_best = np.clip(x_curr + init_delta, min_val, max_val)

        # Project x_best to be within eps of x_original
        delta_from_original = np.clip(x_best - x_original, -eps, eps)
        x_best = np.clip(x_original + delta_from_original, min_val, max_val)

        accepted_count = 0
        half_success = False
        for i_iter in range(iterations):
            p = self._p_selection(0.5, i_iter, iterations)
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h - 1)

            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            delta = x_best - x_curr
            x_curr_window = x_curr[:, center_h:center_h + s, center_w:center_w + s]
            x_best_window = x_best[:, center_h:center_h + s, center_w:center_w + s]

            attempt = 0
            while np.sum(np.abs(np.clip(x_curr_window + delta[:, center_h:center_h + s, center_w:center_w + s], min_val, max_val) - x_best_window) < 1e-7) == c * s * s:
                delta[:, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps], size=[c, 1, 1])
                attempt += 1
                if attempt > 10:
                    break

            x_new = np.clip(x_curr + delta, min_val, max_val)

            # 🔐 Enforce l_infinity constraint w.r.t x_original
            delta_from_original = np.clip(x_new - x_original, -eps, eps)
            x_new = np.clip(x_original + delta_from_original, min_val, max_val)


            response = self._get_comparison_response(
                img_1=x_best, img_2=x_new, prompt=prompt[0] if not half_success else prompt[1]
            )

            eps_curr = np.abs((x_best - x_original).astype(np.int16)).max()


            if response.startswith("1"):
                accepted_count += 1
                x_best = x_new

                x_best_transposed = np.transpose(x_best, (1, 2, 0))
                logprob, test_result = self.get_logprobs_from_vlm(
                    x_best_transposed.astype(np.float32) / 255.0, prompt=test_prompt[0] if not half_success else test_prompt[1]
                )
                if (test_result.startswith("0") and not self.targeted_attack) or (test_result.startswith("1") and self.targeted_attack):
                    half_success = True

                if verbose:
                    console.print(Panel(f"[blue]iter={i_iter}, accepted_count={accepted_count} / {i_iter+1}, eps={eps_curr},\nget_comparison_response={response}\n \nget_logprobs_from_vlm={test_result}\n [/blue]", 
                                    title=f"Black-box wo logprob, success rate: {self.attack_success_rate}"))
                logprob, test_result = self.get_logprobs_from_vlm(
                    x_best_transposed.astype(np.float32) / 255.0, prompt=test_prompt[2]
                )
                if (test_result.startswith("0") and not self.targeted_attack) or (test_result.startswith("1") and self.targeted_attack):
                    attack_success = True
                    break

            if i_iter % 100 == 0:
                console.print(Panel(f"[green]iter={i_iter}, accepted_count={accepted_count} / {i_iter+1}, eps={eps_curr} \n [/green]", 
                                title=f"Black-box wo logprob, success rate: {self.attack_success_rate}"))

        self.gen_img += 1

        return x_best, attack_success


    def black_box_attack_wo_logprob(self, source_x, x_original, verbose=False, iterations=2000, prompt="", test_prompt="", eps=8, **kwargs):
        """
        Generic black-box attack without logprobs using comparative responses.
        Supports both targeted and non-targeted attacks.
        
        Args:
            source_x: Starting point (could be adversarial) to refine attack
            x_original: The true original image (perturbation measured w.r.t. this)
            iterations: Number of attack iterations  
            prompt: Comparison prompt for the model
            test_prompt: Testing prompt to evaluate success
            eps: Maximum perturbation size in uint8 range [0,255]
            **kwargs: Additional model-specific parameters
            
        Returns:
            tuple: (best_adversarial_example, attack_success)
        """
        if not hasattr(self, '_get_comparison_response'):
            raise NotImplementedError("Subclass must implement _get_comparison_response method")
            
        attack_success = False        

        # Prepare images
        x_original = self._prepare_image_for_attack(x_original)
        source_x = self._prepare_image_for_attack(source_x)
        x_curr = source_x.copy()

        np.random.seed(0)
        min_val, max_val = 0, 255
        c, h, w = x_curr.shape
        n_features = c * h * w
        assert isinstance(eps, int), "eps must be an integer"

        # init_delta = np.random.choice([-eps, eps], size=[c, 1, w])
        #! set init_delta to 0
        init_delta = np.zeros([c, 1, w]) if self.targeted_attack else np.random.choice([-eps, eps], size=[c, 1, w])
        x_best = np.clip(x_curr + init_delta, min_val, max_val)

        # Project x_best to be within eps of x_original
        delta_from_original = np.clip(x_best - x_original, -eps, eps)
        x_best = np.clip(x_original + delta_from_original, min_val, max_val)

        accepted_count = 0


        for i_iter in range(iterations):
            p = self._p_selection(0.5, i_iter, iterations)
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h - 1)

            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            delta = x_best - x_curr
            x_curr_window = x_curr[:, center_h:center_h + s, center_w:center_w + s]
            x_best_window = x_best[:, center_h:center_h + s, center_w:center_w + s]

            attempt = 0
            while np.sum(np.abs(np.clip(x_curr_window + delta[:, center_h:center_h + s, center_w:center_w + s], min_val, max_val) - x_best_window) < 1e-7) == c * s * s:
                delta[:, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps], size=[c, 1, 1])
                attempt += 1
                if attempt > 10:
                    break

            x_new = np.clip(x_curr + delta, min_val, max_val)

            # 🔐 Enforce l_infinity constraint w.r.t x_original
            delta_from_original = np.clip(x_new - x_original, -eps, eps)
            x_new = np.clip(x_original + delta_from_original, min_val, max_val)


            response = self._get_comparison_response(
                img_1=x_best, img_2=x_new, prompt=prompt
            )

            eps_curr = np.abs((x_best - x_original).astype(np.int16)).max()

            # self._save_comparison_plot(x_original, x_curr, x_new, filename=f"adv_clean.png")
            # import pdb; pdb.set_trace()
            # console.print(f"[red]response: {response}[/red]")
            # prob = 0

            # for i in range(5):
            #     x_best_transposed = np.transpose(x_best, (1, 2, 0))
            #     logprob_1, test_result = self.get_logprobs_from_vlm(
            #         x_best_transposed.astype(np.float32) / 255.0, prompt=test_prompt
            #     )
            #     x_new_transposed = np.transpose(x_new, (1, 2, 0))
            #     logprob_2, test_result = self.get_logprobs_from_vlm(
            #         x_new_transposed.astype(np.float32) / 255.0, prompt=test_prompt
            #     )

            #     prob += np.exp(logprob_1) - np.exp(logprob_2)
            # prob /= 5
            # print(f"iter={i_iter}, prob={prob}, accepted?: {response.startswith('1')}")
            # self.score_list.append([prob, response.startswith("1")])
            # np.save(f"plot/setting_a2/{self.gen_img}.npy", np.array(self.score_list))

            if response.startswith("1"):
                accepted_count += 1
                # self._save_comparison_plot(x_best, x_curr, x_new, filename=f"plot/adv_clean_{i_iter}.png")
                # import pdb; pdb.set_trace()
                x_best = x_new

                x_best_transposed = np.transpose(x_best, (1, 2, 0))
                logprob, test_result = self.get_logprobs_from_vlm(
                    x_best_transposed.astype(np.float32) / 255.0, prompt=test_prompt
                )

                if verbose:
                    console.print(Panel(f"[blue]iter={i_iter}, accepted_count={accepted_count} / {i_iter+1}, eps={eps_curr},\nget_comparison_response={response}\n \nget_logprobs_from_vlm={test_result}\n [/blue]", 
                                    title=f"Black-box wo logprob, success rate: {self.attack_success_rate}"))
                if (test_result.startswith("0") and not self.targeted_attack) or (test_result.startswith("1") and self.targeted_attack):
                    attack_success = True
                    break

            if i_iter % 100 == 0:
                console.print(Panel(f"[green]iter={i_iter}, accepted_count={accepted_count} / {i_iter+1}, eps={eps_curr} \n [/green]", 
                                title=f"Black-box wo logprob, success rate: {self.attack_success_rate}"))

        self.gen_img += 1

        return x_best, attack_success


    def run_attack(self, image_path, label_name, x_original, verbose=False, attack_type="black_box_w_logprob",source_label=None, **kwargs):
        """
        Convenience method to run any attack type.
        
        Args:
            image_path: Path to input image
            label_name: Target label name
            x_original: Original image
            verbose: Verbose mode
            attack_type: Type of attack to run
            **kwargs: Additional arguments for specific attack methods
            
        Returns:
            tuple: (adversarial_example, attack_success)
        """
        # Validate attack type
        supported_attacks = self._get_supported_attack_types()
        if attack_type not in supported_attacks:
            raise ValueError(f"Attack type '{attack_type}' not supported. Available: {supported_attacks}")
        
        # Load and preprocess image
        x = self._load_and_preprocess_image(image_path)
        # Get attack method
        attack_method = getattr(self, attack_type)
        # Prepare prompts
        if attack_type != "black_box_attack_wo_logprob":
            if self.targeted_attack:
                prompt = f"Does this image contain a {label_name} and doesn't contain a {source_label}? If yes, output 1. If no, output 0. You can only output 1 when the image contains a {label_name} and doesn't contain a {source_label}, otherwise always output 0. Directly output the number first, then explain your reasoning. Always output the number first."
                # prompt = f"Does this image contain a {label_name}? If this image contains a {label_name}, output 1. If this image does not contain a {label_name}, output 0. Directly output the number first, then explain your reasoning. Always output the number first. Example output: '1, because [reason], so this image does contain a {label_name}' or '0, because [reason], so this image does not contain a {label_name}'"

            else:
                prompt = f"Does this image contain a {label_name}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            return attack_method(source_x=x, x_original=x_original, verbose=verbose, prompt=prompt, **kwargs)
        else:
            if self.targeted_attack:
                prompt = f"""Given the two images, which one is more likely to contain a {label_name} and not contain a {source_label}? These two images are roughly the same, but with some small differences. You should think carefully and try to figure out the difference.
                Reply 1 if the second image is more likely to contain a {label_name} and not contain a {source_label}, reply 0 if the first image is more likely to contain a {label_name} and not contain a {source_label}. Always output the number first and then explain why you think which one is more likely to contain a {label_name} and not contain a {source_label}.
                Strictly follow the format below:
                number\n
                The [first / second] image is more likely to contain a {label_name} and not contain a {source_label}, because [reason]. So I think the answer is [number].
                """     
                # prompt_stage1 = f"""Given the two images, which one is less likely to contain a {source_label}? These two images are roughly the same, but with some small differences.
                # Reply 1 if the second image is less likely to contain a {source_label}, 0 otherwise. Always output the number first and then explain why you think which one is less likely to contain a {source_label}. 
                # Strictly follow the format below:
                # <number>
                # <The [first / second] image is less likely to contain a {source_label}, because [reason]. So I think the answer is [number].>
                # """  
                # prompt_stage2 = f"""Given the two images, which one is more likely to contain a {label_name}? These two images are roughly the same, but with some small differences.
                # Reply 1 if the second image is more likely to contain a {label_name}, 0 otherwise. Always output the number first and then explain why you think which one is more likely to contain a {label_name}. 
                # Strictly follow the format below:
                # <number>
                # <The [first / second] image is more likely to contain a {label_name}, because [reason]. So I think the answer is [number].>
                # """  
                # prompt = (prompt_stage1, prompt_stage2)
                # test_prompt_stage1 = f"Does this image contain a {source_label}? If no, output 1. If yes, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
                # test_prompt_stage2 = f"Does this image contain a {label_name}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."

                test_prompt = f"Does this image contain a {label_name} and not contain a {source_label}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
                # test_prompt = (test_prompt_stage1, test_prompt_stage2, test_prompt)
            else:   
                prompt = f"""Given the two images, which one is less likely to contain a {label_name}? These two images are roughly the same, but with some small differences. You should think carefully and try to figure out the difference.
                Reply 1 if the second image is less likely to contain a {label_name}, reply 0 if the first image is less likely to contain a {label_name}. 
                Always output the number first and then explain why you think which one is less likely to contain a {label_name}. 
                Strictly follow the format below:
                number\n
                The [first / second] image is less likely to contain a {label_name}, because reason. So I think the answer is number.
                """
                test_prompt = f"Does this image contain a {label_name}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            return attack_method(source_x=x, x_original=x_original, verbose=verbose, prompt=prompt, test_prompt=test_prompt, **kwargs)
    
    def test_accuracy(self, dataset):
        """
        Test model accuracy on a dataset.
        
        Args:
            dataset: Dataset containing (image_path, label_name) pairs
            
        Returns:
            float: Accuracy as a fraction
        """
        correct = 0
        for img_path, source_label, target_label in dataset:
            x = self._load_and_preprocess_image(img_path)
            if self.targeted_attack:
                prompt = f"Does this image contain a {target_label} and doesn't contain a {source_label}? If yes, output 1. If no, output 0. You can only output 1 when the image contains a {target_label} and doesn't contain a {source_label}, otherwise always output 0. Directly output the number first, then explain your reasoning. Always output the number first."

                # prompt = f"Does this image contain a {target_label}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            else:
                prompt = f"Does this image contain a {source_label}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            logprob, text = self.get_logprobs_from_vlm(x, prompt=prompt)
            label_name = target_label if self.targeted_attack else source_label
            console.print(Panel(f"[yellow]Label: {label_name}, image: {img_path}, Text: {text}, Logprob: {logprob:.4f}[/yellow]", 
                               title="Test Accuracy"))
            if text.startswith("1"):
                correct += 1
        console.print(f"[green]Accuracy: {correct / len(dataset) * 100:.2f}%[/green]")
        return correct / len(dataset) * 100
    
    def evaluate_attack_on_dataset(self, dataset, dataset_original, verbose=False, attack_type="black_box_w_logprob", iterations=1000, eps=8, **kwargs):
        """
        Evaluate attack performance on a dataset.
        
        Args:
            dataset: Dataset containing (image_path, label_name) pairs
            dataset_original: Original dataset containing (image_path, label_name) pairs
            verbose: Verbose mode
            attack_type: Type of attack to perform
            iterations: Number of attack iterations
            eps: Perturbation budget
            **kwargs: Additional attack parameters
            
        Returns:
            dict: Attack evaluation results
        """
        console.print(f"[cyan]Dataset loaded with {len(dataset)} samples[/cyan]")
        
        # Attack configuration
        attack_config = {
            "attack_type": attack_type,
            "iterations": iterations,
            "eps": eps,
            **kwargs
        }
        
        # Track attack results
        attack_results = {
            "total": 0,
            "successful": 0,
            "successful_original": 0,
            "failed": 0,
            "errors": 0,
            "attack_config": attack_config,
            "before_optimization": [],
            "after_optimization": []
        }
        # Attack each sample in the dataset
        for idx, (img_path, source_label, target_label) in enumerate(dataset):
            console.print(f"\n[bold blue]=== Attacking Sample {idx+1}/{len(dataset)}, Image: {img_path}, Label: {source_label}, model: {self.model_name} ===[/bold blue]")
            self.sample_idx = idx + self.args.start_idx
            # Get original response
            x_img = self._load_and_preprocess_image(img_path)
            #!let's set x_original=x_img=transfer_img
            # x_original = x_img
            x_original = self._load_and_preprocess_image(dataset_original[idx][0])
            #! since the dataset is sorted, we can use the index to get the original image
            if self.targeted_attack:
                label_name = target_label
                orig_prompt = f"Does this image contain a {target_label} and doesn't contain a {source_label}? If yes, output 1. If no, output 0. You can only output 1 when the image contains a {target_label} and doesn't contain a {source_label}, otherwise always output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            else:
                label_name = source_label
                orig_prompt = f"Does this image contain a {source_label}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."

            # orig_prompt = f"Does this image contain a {label_name}? If yes, output 1. If no, output 0. Directly output the number first, then explain your reasoning. Always output the number first."
            orig_logprob, orig_text = self.get_logprobs_from_vlm(x_img, prompt=orig_prompt)
            if verbose:
                console.print(Panel(f"[green]Original Response:[/green] {orig_text}\n[green]Original Logprob:[/green] {orig_logprob:.4f}", 
                                    title="Clean Sample"))
            
            # Skip the attack if the prediction is already wrong
            if (orig_text.startswith("0") and not self.targeted_attack) or (orig_text.startswith("1") and self.targeted_attack):
                attack_results["successful"] += 1
                attack_results["successful_original"] += 1
                attack_results["before_optimization"].append(self.sample_idx)

            else:  
                # Run attack
                _, attack_success = self.run_attack(
                    image_path=img_path,
                    label_name=label_name,
                    source_label=source_label,
                    verbose=verbose,
                    x_original=x_original,
                    **attack_config
                )

                if attack_success:
                    attack_results["successful"] += 1
                    attack_results["after_optimization"].append(self.sample_idx)
                else:
                    attack_results["failed"] += 1

            attack_results["total"] += 1
            self.attack_success_rate = str(attack_results["successful"]) + "/" + str(attack_results["total"])
                    

            # Print running statistics
            if attack_results["total"] > 0:
                success_rate = attack_results["successful"] / attack_results["total"] * 100
                console.print(f"[cyan]Running Success Rate: {success_rate:.1f}% ({attack_results['successful']}/{attack_results['total']})[/cyan]")
        
        # Calculate final success rate
        if attack_results["total"] > 0:
            attack_results["success_rate"] = attack_results["successful"] / attack_results["total"]
        else:
            attack_results["success_rate"] = 0.0
            
        # Print final results
        self._print_attack_results(attack_results)
        
        return attack_results
    
    def _print_attack_results(self, results):
        """Print formatted attack evaluation results."""
        console.print("\n" + "="*60)
        console.print("[bold cyan]FINAL ATTACK RESULTS[/bold cyan]")
        console.print(f"[green]Total Samples Attacked: {results['total']}[/green]")
        console.print(f"[green]Successful Attacks: {results['successful']}[/green]") 
        console.print(f"[green]Successful Original Attacks: {results['successful_original']}[/green]")
        console.print(f"[red]Failed Attacks: {results['failed']}[/red]")
        console.print(f"[orange1]Errors: {results['errors']}[/orange1]")
        console.print(f"[blue]Before Optimization: {results['before_optimization']}[/blue]")
        console.print(f"[blue]After Optimization: {results['after_optimization']}[/blue]")
        if results["total"] > 0:
            console.print(f"[bold green]Final Attack Success Rate: {results['success_rate']*100:.1f}%[/bold green]")
        
        console.print("="*60) 

