"""
ReID Embedding Extractor Module
Appearance-based person re-identification using ResNet50
"""

import cv2
import numpy as np
from log import log_with_timestamp

# Optional imports (loaded only when needed)
try:
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingExtractor:
    """
    ReID Appearance Embedding Extractor
    
    Converts person appearances to vectors using ResNet50 backbone
    Provides robust person re-identification capabilities
    """
    
    def __init__(self, device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"):
        """
        Initialize embedding extractor
        
        Args:
            device: 'cuda' or 'cpu' for computation
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available! Install: pip install torch torchvision")
        
        self.device = device
        
        # ResNet50 model (pretrained on ImageNet)
        # Remove final FC layer to extract embedding vectors
        self.model = torchvision.models.resnet50(pretrained=True)
        
        # Remove final FC layer (2048-dimensional embedding)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(device)
        self.model.eval()  # Evaluation mode
        
        # Image preprocessing pipeline (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard ReID dimensions
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        log_with_timestamp(f"ReID Embedding Extractor ready! (Device: {device})", "REID")
    
    def extract_embedding(self, image_crop):
        """
        Extract embedding from cropped image
        
        Args:
            image_crop: NumPy array [H, W, 3] in BGR format
        
        Returns:
            embedding: NumPy array [2048] - appearance vector
        """
        if image_crop is None or image_crop.size == 0:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference (no gradient computation)
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Reshape from [1, 2048, 1, 1] to [2048]
        embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return embedding
    
    def extract_embeddings_batch(self, image_crops):
        """
        Extract embeddings in batch (faster processing)
        
        Args:
            image_crops: List of NumPy arrays
        
        Returns:
            embeddings: List of NumPy arrays [2048]
        """
        if not image_crops:
            return []
        
        embeddings = []
        for crop in image_crops:
            emb = self.extract_embedding(crop)
            embeddings.append(emb)
        
        return embeddings

