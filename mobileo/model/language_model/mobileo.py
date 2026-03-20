from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class mobileoFastConfig(Qwen2Config):
    model_type = "llava_qwen2"


class mobileoFastModel(LlavaMetaModel, Qwen2Model):
    config_class = mobileoFastConfig

    def __init__(self, config: Qwen2Config):
        super(mobileoFastModel, self).__init__(config)

class mobileoFastForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = mobileoFastConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = mobileoFastModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_image: Optional[torch.FloatTensor] = None,
        und_image: Optional[torch.FloatTensor] = None,
        categories: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = True

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        ce_loss = output.loss
        hidden_states = output.hidden_states
        logits = output.logits
        img_hidden_states = hidden_states

        assert (
            latents is not None
        ), "Currently we only support image loss when latents is None"

        weighting_scheme = "uniform"
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (
            u * self.get_model().noise_scheduler.config.num_train_timesteps
        ).long()
        timesteps = (
            self.get_model()
            .noise_scheduler.timesteps[indices]
            .to(device=latents.device)
        )
        sigmas = self.get_sigmas(
            timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
        )

        noise = torch.randn_like(latents, device=latents.device)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        diffusion_pred = (
            self.get_model()
            .dit(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=self.get_model().diffusion_connector(
                    img_hidden_states
                ),
                encoder_attention_mask=attention_mask,
            )
            .sample
        )
        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
        diff_loss = torch.mean(
            (
                weighting.float() * (diffusion_pred.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            1,
        )
        diff_loss = diff_loss.mean()

        ce_weight = 0.2
        diff_weight = 1.0

        total_loss = ce_weight * ce_loss + diff_weight * diff_loss

        print(f"diff_loss: {diff_loss}, ce_loss: {ce_loss}")

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


AutoConfig.register("llava_qwen2", mobileoFastConfig)
AutoModelForCausalLM.register(mobileoFastConfig, mobileoFastForCausalLM)
