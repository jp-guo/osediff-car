import os
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'diff-car/diffusion'))
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
import cv2
import torchvision.models as models
import pyiqa
from my_utils.vaehook import VAEHook, perfcount
from models.qf_encoder import QFEncoderUNet


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
    if args.train_decoder:
        l_target_modules_decoder = []
        l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
        for n, p in vae.named_parameters():
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if pattern in n and ("decoder" in n):
                    l_target_modules_decoder.append(n.replace(".weight", ""))
                elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                    l_target_modules_decoder.append(n.replace(".weight", ""))

        lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                       target_modules=l_target_modules_decoder)
        vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")


    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    if args.train_decoder:
        return vae, l_target_modules_encoder, l_target_modules_decoder

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
              "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class QF_OSEDiff(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path,
                                                          subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        # self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
        #                      decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        # if self.args.merge_and_unload_lora:
        #     print(f'===> MERGE LORA <===')
        #     self.vae = self.vae.merge_and_unload()
        #     self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        self.qf_encoder = QFEncoderUNet().to("cuda")

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()

        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.qf_encoder.train()

    def set_eval(self):
        self.qf_encoder.eval()

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        if self.args.train_decoder:
            vae_lora_conf_decoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                               target_modules=model["vae_lora_decoder_modules"])
            self.vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])
        if self.args.train_decoder:
            self.vae.set_adapter(['default_decoder'])

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def eval(self, lq, prompt, qf):
        # lq, qf = lq.type(self.weight_dtype), qf.type(self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.type(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        qf_embeddings = self.qf_encoder(lq_latent, qf)
        for i in range(len(qf_embeddings)):
            qf_embeddings[i] = qf_embeddings[i].type(self.weight_dtype)
        down_block_additional_residuals = qf_embeddings[:-1]
        mid_block_additional_residual = qf_embeddings[-1]

        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds,
                               down_block_additional_residuals=down_block_additional_residuals,
                               mid_block_additional_residual=mid_block_additional_residual).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised.type(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image


    def forward(self, c_t, batch, qf):
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor    # b, 4, 16, 16

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])     # 1, 77, 1024
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        qf_embeddings = self.qf_encoder(encoded_control, qf=qf)

        for i in range(len(qf_embeddings)):
            qf_embeddings[i] = qf_embeddings[i].type(self.weight_dtype)
        down_block_additional_residuals = qf_embeddings[:-1]
        mid_block_additional_residual = qf_embeddings[-1]

        model_pred = self.unet(encoded_control, self.timesteps,
                               encoder_hidden_states=prompt_embeds.to(torch.float32),
                               down_block_additional_residuals=down_block_additional_residuals,
                               mid_block_additional_residual=mid_block_additional_residual).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control,
                                               return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        # breakpoint()

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds


    def _init_tiled_vae(self,
                        encoder_tile_size=256,
                        decoder_tile_size=256,
                        fast_decoder=False,
                        fast_encoder=False,
                        color_fix=False,
                        vae_to_gpu=True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x
            in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for
            y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

    def save_model(self, outf):
        torch.save(self.qf_encoder.state_dict(), outf)


class QF_OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path,
                                                          subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
                             decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        self.qf_encoder = QFEncoderUNet().to("cuda")
        self.qf_encoder.load_state_dict(torch.load(args.qf_encoder_path))

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                           target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        if self.args.train_decoder:
            vae_lora_conf_decoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian",
                                               target_modules=model["vae_lora_decoder_modules"])
            self.vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])
        if self.args.train_decoder:
            self.vae.set_adapter(['default_decoder'])

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt, qf):
        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        qf_embeddings = self.qf_encoder(lq_latent, qf)
        for i in range(len(qf_embeddings)):
            qf_embeddings[i] = qf_embeddings[i].type(self.weight_dtype)
        down_block_additional_residuals = qf_embeddings[:-1]
        mid_block_additional_residual = qf_embeddings[-1]

        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds,
                               down_block_additional_residuals=down_block_additional_residuals,
                               mid_block_additional_residual=mid_block_additional_residual).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def _init_tiled_vae(self,
                        encoder_tile_size=256,
                        decoder_tile_size=256,
                        fast_decoder=False,
                        fast_encoder=False,
                        color_fix=False,
                        vae_to_gpu=True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x
            in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for
            y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))