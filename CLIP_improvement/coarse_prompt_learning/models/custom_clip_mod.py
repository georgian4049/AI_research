import torch
import torch.nn as nn
from models.prompt_learner import PromptLearner
from models.clip_model import load_clip_to_cpu
from torch.nn import functional as F

class PromptGuidedImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.cross_attention_fine = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.cross_attention_coarse = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        # Add a projection layer to match dimensions
        self.proj = nn.Linear(self.clip_visual.output_dim, 512) if hasattr(self.clip_visual, 'output_dim') else nn.Linear(512, 512)

    def forward(self, image, fine_prompts, coarse_prompts):
        # Get base image features from CLIP visual encoder
        base_features = self.clip_visual(image)  # (batch_size, embed_dim)
        
        # Project to match attention dimensions if needed
        base_features = self.proj(base_features)  # (batch_size, 512)
        
        # Expand dimensions for attention
        base_features = base_features.unsqueeze(0)  # (1, batch_size, 512)
        
        # Debug: Print shapes
        print(f"base_features shape: {base_features.shape}")
        print(f"fine_prompts shape: {fine_prompts.shape}")
        print(f"coarse_prompts shape: {coarse_prompts.shape}")
        
        # Ensure fine_prompts and coarse_prompts have the correct shape
        if fine_prompts.dim() == 4:
            fine_prompts = fine_prompts.squeeze(0)  # Remove extra dimension if present
        if coarse_prompts.dim() == 4:
            coarse_prompts = coarse_prompts.squeeze(0)  # Remove extra dimension if present
        
        # Apply cross-attention with fine prompts
        fine_prompts = fine_prompts.permute(1, 0, 2)  # (n_fine_ctx, batch_size, embed_dim)
        attended_features_fine, _ = self.cross_attention_fine(base_features, fine_prompts, fine_prompts)
        
        # Apply cross-attention with coarse prompts
        coarse_prompts = coarse_prompts.permute(1, 0, 2)  # (n_coarse_ctx, batch_size, embed_dim)
        attended_features_coarse, _ = self.cross_attention_coarse(attended_features_fine, coarse_prompts, coarse_prompts)
        
        # Squeeze back to original shape
        image_embedding = attended_features_coarse.squeeze(0)  # (batch_size, 512)
        
        return image_embedding

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, coarse_classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, coarse_classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_coarse_prompts = self.prompt_learner.tokenized_coarse_prompts
        self.image_encoder = PromptGuidedImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # Add base visual encoder for initial prompt generation
        self.base_visual = clip_model.visual

    def forward(self, image, fine_labels=None, coarse_labels=None):
        tokenized_prompts = self.tokenized_prompts
        tokenized_coarse_prompts = self.tokenized_coarse_prompts
        logit_scale = self.logit_scale.exp()

        # Get initial image features for prompt learning
        base_image_features = self.base_visual(image.type(self.dtype))
        base_image_features = base_image_features / base_image_features.norm(dim=-1, keepdim=True)
        
        # Generate fine and coarse prompts using base features
        fine_prompts, coarse_prompts = self.prompt_learner(base_image_features)

        # Debug: Print shapes of generated prompts
        print(f"Generated fine_prompts shape: {fine_prompts.shape}")
        print(f"Generated coarse_prompts shape: {coarse_prompts.shape}")

        # Get enhanced image features using prompt-guided encoder
        image_features = self.image_encoder(image.type(self.dtype), fine_prompts, coarse_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute fine-grained logits
        text_features_fine = self.text_encoder(fine_prompts, tokenized_prompts)
        text_features_fine = text_features_fine / text_features_fine.norm(dim=-1, keepdim=True)
        fine_logits = logit_scale * image_features @ text_features_fine.t()

        # Compute coarse-grained logits
        text_features_coarse = self.text_encoder(coarse_prompts, tokenized_coarse_prompts)
        text_features_coarse = text_features_coarse / text_features_coarse.norm(dim=-1, keepdim=True)
        coarse_logits = logit_scale * image_features @ text_features_coarse.t()

        if self.training:
            assert fine_labels is not None and coarse_labels is not None, "Labels must be provided during training"
            loss_fine = F.cross_entropy(fine_logits, fine_labels)
            loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)
            total_loss = loss_fine + loss_coarse
            return total_loss, loss_fine, loss_coarse

        return fine_logits, coarse_logits