import torch
import torch.nn as nn
from models.prompt_learner import PromptLearner
from models.clip_model import load_clip_to_cpu
from torch.nn import functional as F

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
    def __init__(self, cfg, classnames, coarse_classnames, fine_to_coarse, clip_model):
        super().__init__()
        self.model_type = cfg['MODEL']['TYPE']
        self.prompt_learner = PromptLearner(cfg, classnames, coarse_classnames, fine_to_coarse, clip_model)
        self.fine_to_coarse = fine_to_coarse
        self.coarse_classnames = coarse_classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_coarse_prompts = self.prompt_learner.tokenized_coarse_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.coarse_loss_weight = cfg.get('MODEL', {}).get('COARSE_LOSS_WEIGHT', 0.5)  # Weight for loss_coarse

    def forward(self, image, fine_labels=None, coarse_labels=None):
        tokenized_prompts = self.tokenized_prompts
        tokenized_coarse_prompts = self.tokenized_coarse_prompts
        logit_scale = self.logit_scale.exp()

        # Extract image features
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Generate prompts
        prompts = self.prompt_learner(image_features)
        fine_prompts = prompts[0] if isinstance(prompts, tuple) else prompts
        coarse_prompts = prompts[1] if isinstance(prompts, tuple) else None

        # Compute fine-grained logits
        fine_logits = []
        for pts_i, imf_i in zip(fine_prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            fine_logits.append(l_i)
        fine_logits = torch.stack(fine_logits).to(torch.float32)

        # Compute coarse-grained logits (only for fine_coarse_separate and fine_context)
        coarse_logits = None
        if self.model_type in ['fine_coarse_separate', 'fine_context'] and coarse_prompts is not None:
            coarse_logits = []
            for pts_i, imf_i in zip(coarse_prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_coarse_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                coarse_logits.append(l_i)
            coarse_logits = torch.stack(coarse_logits).to(torch.float32)

        # Compute losses during training
        if self.training:
            assert fine_labels is not None, "Fine labels must be provided during training"
            loss_fine = F.cross_entropy(fine_logits, fine_labels)
            loss_coarse = 0.0
            if self.model_type in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] and coarse_labels is not None:
                # Map fine logits to coarse logits for fine_coarse_concat and fine_coarse_concat_cross
                if self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross']:
                    coarse_logits = torch.zeros(fine_logits.size(0), len(self.coarse_classnames), device=fine_logits.device)
                    for fine_idx, coarse_cls in enumerate(self.fine_to_coarse.values()):
                        coarse_idx = self.coarse_classnames.index(coarse_cls)
                        coarse_logits[:, coarse_idx] += fine_logits[:, fine_idx]
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)
                elif coarse_logits is not None:
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)
            total_loss = loss_fine + self.coarse_loss_weight * loss_coarse
            return total_loss, loss_fine, loss_coarse

        # Return logits based on model type
        if self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross', 'fine_only', 'plain_fine']:
            return fine_logits
        return fine_logits, coarse_logits