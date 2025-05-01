import torch
import torch.nn as nn
import clip

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, coarse_class, fine_to_coarse, clip_model):
        super().__init__()
        self.model_type = cfg['MODEL']['TYPE']
        n_fine_cls = len(classnames)
        n_coarse_cls = len(coarse_class) if coarse_class else 0
        n_fine_ctx = 4 if self.model_type not in ['plain_fine', 'plain_fine_coarse'] else 0
        n_coarse_ctx = 2 if self.model_type in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else 0
        ctx_fine_init = "a photo of a" if n_fine_ctx > 0 else ""
        ctx_coarse_init = "of type" if n_coarse_ctx > 0 else ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # Initialize fine context vectors
        if ctx_fine_init and n_fine_ctx > 0:
            prompt = clip.tokenize(ctx_fine_init)
            with torch.no_grad():
                fine_embedding = clip_model.token_embedding(prompt).to(dtype)
            ctx_fine_vectors = fine_embedding[0, 1:1 + n_fine_ctx, :]
            prompt_fine_prefix = ctx_fine_init
        elif n_fine_ctx > 0:
            ctx_fine_vectors = torch.empty(n_fine_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_fine_vectors, std=0.02)
            prompt_fine_prefix = " ".join(["X"] * n_fine_ctx)
        else:
            ctx_fine_vectors = None
            prompt_fine_prefix = ""

        # Initialize coarse context vectors
        ctx_coarse_vectors = None
        prompt_coarse_prefix = ""
        if self.model_type in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] and n_coarse_ctx > 0:
            if ctx_coarse_init:
                prompt = clip.tokenize(ctx_coarse_init)
                with torch.no_grad():
                    coarse_embedding = clip_model.token_embedding(prompt).to(dtype)
                ctx_coarse_vectors = coarse_embedding[0, 1:1 + n_coarse_ctx, :]
                prompt_coarse_prefix = ctx_coarse_init
            else:
                ctx_coarse_vectors = torch.empty(n_coarse_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_coarse_vectors, std=0.02)
                prompt_coarse_prefix = " ".join(["X"] * n_coarse_ctx)

        # Learnable parameters
        self.fine_ctx = nn.Parameter(ctx_fine_vectors) if ctx_fine_vectors is not None else None
        self.coarse_ctx = nn.Parameter(ctx_coarse_vectors) if ctx_coarse_vectors is not None else None

        # Meta network for image feature transformation
        self.meta_net = None
        if self.model_type in ['fine_only', 'fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross']:
            self.meta_net = nn.Sequential(
                nn.Linear(vis_dim, ctx_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ctx_dim, ctx_dim)
            )

        # Cross-attention for fine_context and fine_coarse_concat_cross
        self.cross_attention = None
        if self.model_type in ['fine_context', 'fine_coarse_concat_cross']:
            self.cross_attention = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=1)

        # Convert to half precision if needed
        if cfg['MODEL']['PRECISION'] == "fp16":
            if self.meta_net is not None:
                self.meta_net.half()
            if self.cross_attention is not None:
                self.cross_attention.half()

        # Prepare prompts based on model type
        if self.model_type == 'plain_fine':
            prompts = ["a photo of a " + name.replace("_", " ") + "." for name in classnames]
        elif self.model_type == 'plain_fine_coarse':
            prompts = [
                f"a photo of a {name.replace('_', ' ')} of type {fine_to_coarse[name]}." 
                for name in classnames
            ]
        elif self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross']:
            prompts = [
                f"{prompt_fine_prefix} {name.replace('_', ' ')} {prompt_coarse_prefix} {fine_to_coarse[name]}." 
                for name in classnames
            ]
        else:
            prompts = [prompt_fine_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        
        coarse_prompts = []
        if self.model_type in ['fine_coarse_separate', 'fine_context'] and coarse_class:
            coarse_prompts = [prompt_coarse_prefix + " " + name + "." for name in coarse_class]

        # Tokenize prompts
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_coarse_prompts = torch.cat([clip.tokenize(p) for p in coarse_prompts]) if coarse_prompts else None

        # Register buffers for prefix and suffix tokens
        if self.model_type in ['fine_only', 'fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross']:
            with torch.no_grad():
                fine_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("token_fine_prefix", fine_embedding[:, :1, :])  # SOS
            if self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross']:
                self.register_buffer("token_fine_suffix", fine_embedding[:, 1 + n_fine_ctx:2 + n_fine_ctx, :])  # Fine CLS
                self.register_buffer("token_coarse_suffix", fine_embedding[:, 2 + n_fine_ctx + n_coarse_ctx:, :])  # Coarse CLS, EOS
            else:
                self.register_buffer("token_fine_suffix", fine_embedding[:, 1 + n_fine_ctx:, :])  # CLS, EOS
                self.register_buffer("token_coarse_suffix", None)
            if self.model_type in ['fine_coarse_separate', 'fine_context'] and coarse_prompts:
                with torch.no_grad():
                    coarse_embedding = clip_model.token_embedding(tokenized_coarse_prompts).type(dtype)
                self.register_buffer("token_coarse_prefix", coarse_embedding[:, :1, :])  # SOS
                self.register_buffer("token_coarse_suffix", coarse_embedding[:, 1 + n_coarse_ctx:, :])  # CLS, EOS
            else:
                self.register_buffer("token_coarse_prefix", None)
        else:
            self.register_buffer("token_fine_prefix", None)
            self.register_buffer("token_fine_suffix", None)
            self.register_buffer("token_coarse_prefix", None)
            self.register_buffer("token_coarse_suffix", None)

        self.n_fine_cls = n_fine_cls
        self.n_fine_ctx = n_fine_ctx
        self.n_coarse_cls = n_coarse_cls
        self.n_coarse_ctx = n_coarse_ctx
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_coarse_prompts = tokenized_coarse_prompts
        self.fine_to_coarse = fine_to_coarse

    def construct_prompts(self, ctx, prefix, suffix, coarse_ctx=None, coarse_suffix=None, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
            if coarse_suffix is not None:
                coarse_suffix = coarse_suffix[label]
        if coarse_ctx is not None and coarse_suffix is not None:
            prompts = torch.cat([prefix, ctx, suffix, coarse_ctx, coarse_suffix], dim=1)
        else:
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        if self.model_type in ['plain_fine', 'plain_fine_coarse']:
            return self.tokenized_prompts.unsqueeze(0).expand(im_features.size(0), -1, -1)

        batch_size = im_features.size(0)
        ctx_dim = self.fine_ctx.size(-1) if self.fine_ctx is not None else None

        # Generate transformed image features
        if self.meta_net is not None:
            image_features_transformed = self.meta_net(im_features)
            image_features_transformed = image_features_transformed.unsqueeze(1)
        else:
            image_features_transformed = None

        # Prepare fine context
        fine_ctx = None
        if self.fine_ctx is not None:
            fine_ctx = self.fine_ctx.unsqueeze(0).expand(batch_size, -1, -1)
            if image_features_transformed is not None:
                fine_ctx = fine_ctx + image_features_transformed

        # Prepare coarse context
        coarse_ctx = None
        if self.model_type in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] and self.coarse_ctx is not None:
            coarse_ctx = self.coarse_ctx.unsqueeze(0).expand(batch_size, -1, -1)
            if image_features_transformed is not None:
                coarse_ctx = coarse_ctx + image_features_transformed

        # Apply cross-attention
        if self.model_type in ['fine_context', 'fine_coarse_concat_cross'] and self.cross_attention is not None:
            fine_ctx = fine_ctx.permute(1, 0, 2)  # (n_fine_ctx, batch, ctx_dim)
            coarse_ctx = coarse_ctx.permute(1, 0, 2)  # (n_coarse_ctx, batch, ctx_dim)
            fine_ctx, _ = self.cross_attention(fine_ctx, coarse_ctx, coarse_ctx)
            fine_ctx = fine_ctx.permute(1, 0, 2)
            coarse_ctx = coarse_ctx.permute(1, 0, 2)

        # Construct fine-grained prompts
        fine_prompts = []
        if self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross']:
            for ctx_shifted_i, coarse_ctx_shifted_i in zip(fine_ctx, coarse_ctx):
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_fine_cls, -1, -1)
                coarse_ctx_i = coarse_ctx_shifted_i.unsqueeze(0).expand(self.n_fine_cls, -1, -1)
                pts_i = self.construct_prompts(
                    ctx_i, self.token_fine_prefix, self.token_fine_suffix,
                    coarse_ctx_i, self.token_coarse_suffix
                )
                fine_prompts.append(pts_i)
        else:
            for ctx_shifted_i in fine_ctx:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_fine_cls, -1, -1)
                pts_i = self.construct_prompts(ctx_i, self.token_fine_prefix, self.token_fine_suffix)
                fine_prompts.append(pts_i)
        fine_prompts = torch.stack(fine_prompts)

        # Construct coarse-grained prompts
        coarse_prompts = None
        if self.model_type in ['fine_coarse_separate', 'fine_context'] and coarse_ctx is not None:
            coarse_prompts = []
            for ctx_shifted_i in coarse_ctx:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_coarse_cls, -1, -1)
                pts_i = self.construct_prompts(ctx_i, self.token_coarse_prefix, self.token_coarse_suffix)
                coarse_prompts.append(pts_i)
            coarse_prompts = torch.stack(coarse_prompts)

        if self.model_type in ['fine_coarse_concat', 'fine_coarse_concat_cross', 'fine_only']:
            return fine_prompts
        return fine_prompts, coarse_prompts