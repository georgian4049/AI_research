import torch
import torch.nn as nn
import clip

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, coarse_class, clip_model):
        super().__init__()
        n_fine_cls = len(classnames)
        n_coarse_cls = len(coarse_class)
        n_fine_ctx = 4  # Number of fine context tokens
        n_coarse_ctx = 4  # Number of coarse context tokens
        ctx_fine_init = "a class of a"
        ctx_coarse_init = "of type having heirarchy"
        dtype = clip_model.dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # Initialize fine and coarse context vectors
        if ctx_fine_init:
            prompt = clip.tokenize(ctx_fine_init)
            with torch.no_grad():
                fine_embedding = clip_model.token_embedding(prompt).to(dtype)
            ctx_fine_vectors = fine_embedding[0, 1:1 + n_fine_ctx, :]
            prompt_fine_prefix = ctx_fine_init
        else:
            ctx_fine_vectors = torch.empty(n_fine_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_fine_vectors, std=0.02)
            prompt_fine_prefix = " ".join(["X"] * n_fine_ctx)

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

        # print(f'Initial Fine context: "{prompt_fine_prefix}"')
        # print(f"Number of fine context words (tokens): {n_fine_ctx}")
        # print(f'Initial Coarse context: "{prompt_coarse_prefix}"')
        # print(f"Number of coarse context words (tokens): {n_coarse_ctx}")

        # Learnable parameters
        self.fine_ctx = nn.Parameter(ctx_fine_vectors)  # (n_fine_ctx, ctx_dim)
        self.coarse_ctx = nn.Parameter(ctx_coarse_vectors)  # (n_coarse_ctx, ctx_dim)

        # Meta network to generate bias from image features
        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, ctx_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim, ctx_dim)
        )

        # Cross-attention for hierarchical dependency
        self.cross_attention = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=1)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.Sigmoid()
        )

        if cfg['MODEL']['PRECISION'] == "fp16":
            self.meta_net.half()
            self.cross_attention.half()
            self.gate.half()

        # Tokenized prompts
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_fine_prefix + " " + name + "." for name in classnames]
        prompts_coarse = [prompt_coarse_prefix + " " + name + "." for name in coarse_class]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_fine_cls, n_tkn)
        tokenized_coarse_prompts = torch.cat([clip.tokenize(p) for p in prompts_coarse])  # (n_coarse_cls, n_tkn)

        with torch.no_grad():
            fine_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            coarse_embedding = clip_model.token_embedding(tokenized_coarse_prompts).type(dtype)

        # Register buffers for prefix and suffix tokens
        self.register_buffer("token_fine_prefix", fine_embedding[:, :1, :])  # SOS
        self.register_buffer("token_fine_suffix", fine_embedding[:, 1 + n_fine_ctx:, :])  # CLS, EOS
        self.register_buffer("token_coarse_prefix", coarse_embedding[:, :1, :])  # SOS
        self.register_buffer("token_coarse_suffix", coarse_embedding[:, 1 + n_coarse_ctx:, :])  # CLS, EOS

        self.n_fine_cls = n_fine_cls
        self.n_fine_ctx = n_fine_ctx
        self.n_coarse_cls = n_coarse_cls
        self.n_coarse_ctx = n_coarse_ctx
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_coarse_prompts = tokenized_coarse_prompts
        self.name_lens = [len(clip.tokenize(name)) for name in classnames]

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        batch_size = im_features.size(0)  # Get batch size
        ctx_dim = self.fine_ctx.size(-1)  # Get context dimension
        # Generate transformed image features
        image_features_transformed = self.meta_net(im_features)  # (batch, ctx_dim)
        image_features_transformed = image_features_transformed.unsqueeze(1)  # (batch, 1, ctx_dim)

        # Prepare fine and coarse contexts
        fine_ctx = self.fine_ctx.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_fine_ctx, ctx_dim)
        coarse_ctx = self.coarse_ctx.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_coarse_ctx, ctx_dim)

        # Debug: Print shapes
        # print(f"fine_ctx shape: {fine_ctx.shape}")
        # print(f"coarse_ctx shape: {coarse_ctx.shape}")
        # print(f"image_features_transformed shape: {image_features_transformed.shape}")

        # Fuse image features with fine and coarse contexts
        fine_ctx = fine_ctx + image_features_transformed
        coarse_ctx = coarse_ctx + image_features_transformed

        # Apply cross-attention for hierarchical dependency
        fine_ctx = fine_ctx.permute(1, 0, 2)  # (n_fine_ctx, batch, ctx_dim)
        coarse_ctx = coarse_ctx.permute(1, 0, 2)  # (n_coarse_ctx, batch, ctx_dim)
        fine_ctx, _ = self.cross_attention(fine_ctx, coarse_ctx, coarse_ctx)  # Fine attends to coarse
        fine_ctx = fine_ctx.permute(1, 0, 2)  # (batch, n_fine_ctx, ctx_dim)
        coarse_ctx = coarse_ctx.permute(1, 0, 2)
        # Apply gating mechanism
        gate_value = self.gate(coarse_ctx)  # (batch, n_coarse_ctx, ctx_dim)
        fine_ctx = fine_ctx + gate_value * coarse_ctx

        # Construct fine-grained prompts
        fine_prompts = []
        for ctx_shifted_i in fine_ctx:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_fine_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, self.token_fine_prefix, self.token_fine_suffix)
            fine_prompts.append(pts_i)
        fine_prompts = torch.stack(fine_prompts)

        # Construct coarse-grained prompts
        coarse_prompts = []
        for ctx_shifted_i in coarse_ctx:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_coarse_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, self.token_coarse_prefix, self.token_coarse_suffix)
            coarse_prompts.append(pts_i)
        coarse_prompts = torch.stack(coarse_prompts)

        return fine_prompts, coarse_prompts
    
    