from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobile_block import MobileConditioningProjector
from .multimodal_llava_encoder.builder import build_vision_tower
from .multimodal_llava_projector.builder import build_vision_projector
from .multimodal_decoder.builder import build_vae, build_sana
from diffusers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.normalization import RMSNorm
import math

from mobileo.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


class DiffusionConnector(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=2304, eps=1e-5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=eps, elementwise_affine=True)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        with torch.no_grad():
            self.norm.weight.fill_(math.sqrt(5.5))

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        print("=" * 20, "Initializing the model", "=" * 20)
        print(config)
        print("=" * 50)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "diffusion_name_or_path"):
            self.dit = build_sana(config)
            self.vae = build_vae(config)
            self.diffusion_connector = MobileConditioningProjector(input_dim=config.hidden_size, hidden_dim=512, output_dim=2304, num_layers=config.vlm_num_layers)
            if hasattr(config, "is_train"):
                if config.is_train:
                    print("FlowMatchEulerDiscreteScheduler is used")
                    self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.diffusion_name_or_path, subfolder="scheduler")
                else:
                    print("DPMSolverMultistepScheduler is used")
                    self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(config.diffusion_name_or_path, subfolder="scheduler")
            else:
                print("FlowMatchEulerDiscreteScheduler is used")
                self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.diffusion_name_or_path, subfolder="scheduler")
            
        

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_sana(self):
        dit = getattr(self, 'dit', None)
        if type(dit) is list:
            dit = dit[0]
        if dit is not None:
            dit.to(self.device)
        return dit

    def get_sana_vae(self):
        vae = getattr(self, 'vae', None)
        if type(vae) is list:
            vae = vae[0]
        if vae is not None:
            vae.to(self.device)
        return vae

    def initialize_vision_modules(self, model_args, fsdp=None):
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        mm_patch_merge_type = model_args.mm_patch_merge_type
        

        if self.get_sana() is None:
            dit = build_sana(model_args)
            if hasattr(model_args, "is_train"):
                if model_args.is_train:
                    print("FlowMatchEulerDiscreteScheduler is used")
                    self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_args.diffusion_name_or_path, subfolder="scheduler")
                else:
                    print("DPMSolverMultistepScheduler is used")
                    self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_args.diffusion_name_or_path, subfolder="scheduler")

            if fsdp is not None and len(fsdp) > 0:
                self.dit = [dit]
            else:
                self.dit = dit
        else:
            if fsdp is not None and len(fsdp) > 0:
                dit = self.dit[0]
            else:
                dit = self.dit
        for p in dit.parameters():
            p.requires_grad = False
                
        if self.get_sana_vae() is None:
            vae = build_vae(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vae = [vae]
            else:
                self.vae = vae
        else:
            if fsdp is not None and len(fsdp) > 0:
                vae = self.vae[0]
            else:
                vae = self.vae
        for p in vae.parameters():
            p.requires_grad = False
    

        if self.get_vision_tower() is None:
            print("=" * 20, "Building vision tower", "=" * 20)
            vision_tower = build_vision_tower(model_args)
            

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        
        
        if getattr(self, 'diffusion_connector', None) is None:
            self.diffusion_connector = MobileConditioningProjector(input_dim=self.config.hidden_size, hidden_dim=512, output_dim=2304, num_layers=model_args.vlm_num_layers)
            
        for p in self.diffusion_connector.parameters():
            p.requires_grad = True
        for p in dit.parameters():
            p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.diffusion_name_or_path = model_args.diffusion_name_or_path
        self.config.is_train = True



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.get_model().get_vision_tower()(pixel_values)
        image_features = self.get_model().mm_projector(image_features.to(self.get_model().mm_projector[0].weight.dtype))
        return image_features


    def get_mm_projector(self):
        return self.get_model().mm_projector


    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images=None, und_images=None
    ):
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1 or self.get_vision_tower() is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        if gen_images is not None:
            vae = self.get_model().get_sana_vae()
            vae_device = vae.device
            prompt_image_embeds = vae.encode(gen_images.to(vae_device)).latent if gen_images is not None else None
            prompt_image_embeds = prompt_image_embeds * vae.config.scaling_factor if prompt_image_embeds is not None else None
            target_image_embeds = torch.clone(prompt_image_embeds).detach()
        else:
            target_image_embeds = None
            

        images = und_images
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.visual(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.visual(images) # [B, image_tokens, hidden_size]


        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_input_ids = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])
                if i < num_images:
                    if cur_image_idx < image_features.shape[0]:
                        cur_image_features = image_features[cur_image_idx]
                    else:
                        cur_image_features = image_features[-1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            cur_new_input_ids = torch.cat(cur_new_input_ids, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids.append(cur_new_input_ids)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_input_ids_padded = torch.full((batch_size, max_len), -300, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device) if len(new_input_ids) > 0 else None
        

        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                new_input_ids_padded[i, :cur_len] = cur_new_input_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, target_image_embeds


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
