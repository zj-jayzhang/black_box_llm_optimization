import os

class SampledImageDataset:
    """
    This class is used to load sampled images from a folder.
    
    Supports three modes:
    1. Clean data: filenames like "golf_ball_123.jpg" -> returns (img_path, label)
    2. Untargeted adversarial: same as clean data -> returns (img_path, label)
    3. Targeted adversarial: filenames like "golf_ball_tennis_ball.jpg" -> returns (img_path, source_label, target_label)
    
    Args:
        targeted_attack (bool): If True, parse filenames as source_target format for targeted attacks.
                               If False, parse filenames as standard label_number format.
    """
    def __init__(self, folder, num_samples=None, start_idx=0, end_idx=None, targeted_attack=False):
        self.folder = folder
        self.targeted_attack = targeted_attack
        self.samples = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(folder, fname)
                
                if targeted_attack:
                    # For targeted attacks: filename like "golf_ball_tennis_ball.jpg"
                    # Extract source and target labels
                    name_without_ext = os.path.splitext(fname)[0]
                    parts = name_without_ext.split('_')
                    if len(parts) >= 2:
                        # Find the split point - assume the last part that forms a valid label is the target
                        # For now, we'll assume the last underscore separates source from target
                        # This assumes target labels don't contain underscores
                        source_parts = parts[:-1]
                        target_part = parts[-1]
                        source_label = '_'.join(source_parts).split('_')[0]
                        target_label = target_part
                        self.samples.append((img_path, source_label, target_label))
                    else:
                        # Fallback: treat as non-targeted
                        label = parts[0] if parts else name_without_ext
                        self.samples.append((img_path, label, None))
                else:
                    # For clean data or untargeted attacks: filename like "golf_ball_123.jpg"
                    label = fname.split('_')[0]
                    self.samples.append((img_path, label))
        if num_samples is not None and end_idx is None:
            self.samples = self.samples[:num_samples]
            print(f"Sampled {num_samples} images from {folder}")
        elif end_idx is not None:
            self.samples = self.samples[start_idx:end_idx]
            print(f"Sampled {len(self.samples)} images from {folder} from {start_idx} to {end_idx}")
        else:
            print(f"Loaded all {len(self.samples)} images from {folder}")

    def __getitem__(self, idx):
        if self.targeted_attack:
            img_path, source_label, target_label = self.samples[idx]
            return img_path, source_label, target_label
        else:
            img_path, label_name = self.samples[idx]
            return img_path, label_name, None

    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        for i in range(len(self.samples)):
            yield self[i]

