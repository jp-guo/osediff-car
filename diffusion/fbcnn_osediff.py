import os
import sys

# sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.expanduser('~'), 'diff-car/diffusion'))
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
# from models.unet_2d_condition import UNet2DConditionModel
from models.unet_2d_fbcnn import UNet2DConditionModel
from peft import LoraConfig
import cv2
import torchvision.models as models
import pyiqa
from my_utils.vaehook import VAEHook, perfcount
from my_utils.jpeg_torch import jpeg_encode
from diffusion.models.fbcnn_module import QFNet


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
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", strict=False)
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


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                          subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        if args.train_decoder:
            self.vae, self.lora_vae_modules_encoder, self.lora_vae_modules_decoder = initialize_vae(self.args)
        else:
            self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(
            self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.unet.qfnet = QFNet(in_c=4)
        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n or "qfnet" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n or "qfnet" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = False

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
    def eval(self, lq, prompt, qf_gt):
        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred, qf = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds, qf_gt=qf_gt)
        model_pred = model_pred.sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def forward(self, c_t, batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor    # b, 4, 16, 16

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])     # 1, 77, 1024
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        qf_gt = batch['qf'].to("cuda").reshape(-1, 1)
        model_pred, qf = self.unet(encoded_control, self.timesteps,
                               encoder_hidden_states=prompt_embeds.to(torch.float32), qf_gt=qf_gt)
        model_pred = model_pred.sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control,
                                               return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, qf

    def save_model(self, outf):
        sd = {}
        if self.args.train_decoder:
            sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["qfnet"] = {k: v for k, v in self.unet.qfnet.state_dict().items()}
        torch.save(sd, outf)


class OSEDiff_test(torch.nn.Module):
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

        self.unet.qfnet = QFNet()
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

        self.unet.qfnet.load_state_dict(model["qfnet"])

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
    def forward(self, lq, prompt):
        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        # breakpoint()
        ## add tile function
        _, _, h, w = lq_latent.size()
        # breakpoint()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        model_pred, qf = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds)
        model_pred = model_pred.sample
        # breakpoint()
        # if h * w <= tile_size * tile_size:
        #     # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
        #     model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        # else:
        #     print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
        #     # tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        #     tile_size = min(tile_size, min(h, w))
        #     tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        #
        #     grid_rows = 0
        #     cur_x = 0
        #     while cur_x < lq_latent.size(-1):
        #         cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        #         grid_rows += 1
        #
        #     grid_cols = 0
        #     cur_y = 0
        #     while cur_y < lq_latent.size(-2):
        #         cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        #         grid_cols += 1
        #
        #     input_list = []
        #     noise_preds = []
        #     for row in range(grid_rows):
        #         noise_preds_row = []
        #         for col in range(grid_cols):
        #             if col < grid_cols - 1 or row < grid_rows - 1:
        #                 # extract tile from input image
        #                 ofs_x = max(row * tile_size - tile_overlap * row, 0)
        #                 ofs_y = max(col * tile_size - tile_overlap * col, 0)
        #                 # input tile area on total image
        #             if row == grid_rows - 1:
        #                 ofs_x = w - tile_size
        #             if col == grid_cols - 1:
        #                 ofs_y = h - tile_size
        #
        #             input_start_x = ofs_x
        #             input_end_x = ofs_x + tile_size
        #             input_start_y = ofs_y
        #             input_end_y = ofs_y + tile_size
        #
        #             # input tile dimensions
        #             input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
        #             input_list.append(input_tile)
        #
        #             if len(input_list) == 1 or col == grid_cols - 1:
        #                 input_list_t = torch.cat(input_list, dim=0)
        #                 # predict the noise residual
        #                 model_out = self.unet(input_list_t, self.timesteps,
        #                                       encoder_hidden_states=prompt_embeds.to(self.weight_dtype), ).sample
        #                 input_list = []
        #             noise_preds.append(model_out)
        #
        #     # Stitch noise predictions for all tiles
        #     noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
        #     contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
        #     # Add each tile contribution to overall latents
        #     for row in range(grid_rows):
        #         for col in range(grid_cols):
        #             if col < grid_cols - 1 or row < grid_rows - 1:
        #                 # extract tile from input image
        #                 ofs_x = max(row * tile_size - tile_overlap * row, 0)
        #                 ofs_y = max(col * tile_size - tile_overlap * col, 0)
        #                 # input tile area on total image
        #             if row == grid_rows - 1:
        #                 ofs_x = w - tile_size
        #             if col == grid_cols - 1:
        #                 ofs_y = h - tile_size
        #
        #             input_start_x = ofs_x
        #             input_end_x = ofs_x + tile_size
        #             input_start_y = ofs_y
        #             input_end_y = ofs_y + tile_size
        #
        #             noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
        #                                                                                           row * grid_cols + col] * tile_weights
        #             contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        #     # Average overlapping areas with more than 1 contributor
        #     noise_pred /= contributors
        #     model_pred = noise_pred
        #
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def guided_forward(self, lq, prompt, hq, loss_type, alpha=0.005, bp=1, qf=None):
        # hq can be anything, lq image, hq image, decoded image from fbcnn, etc.
        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        _, _, h, w = lq_latent.size()

        hq_img = (hq * 0.5 + 0.5) * 255.
        hq_img = hq_img[0].permute(1, 2, 0)
        hq_img = hq_img.cpu().numpy()
        hq_img = cv2.cvtColor(hq_img, cv2.COLOR_BGR2GRAY)

        with torch.no_grad():
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample

        sobel_x = cv2.Sobel(hq_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(hq_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_magnitude = torch.tensor(sobel_magnitude, device=lq.device).unsqueeze(0).unsqueeze(0)
        sobel_magnitude /= torch.max(sobel_magnitude)
        # if loss_type == 'dists':
        #     dists_metric = pyiqa.create_metric('dists', device=hq.device, as_loss=True, loss_reduction='sum')
        # elif loss_type == 'ssim':
        #     dists_metric = pyiqa.create_metric('ssim', device=hq.device, as_loss=True, loss_reduction='sum')

        for _ in range(bp):
            x_denoised.requires_grad = True
            x_denoised.retain_grad()
            output_image = self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
            # output_image.retain_grad()
            output_image = output_image.float()
            if loss_type == 'mse':
                # loss = ((1 - sobel_magnitude) * (output_image - hq) ** 2).sum()
                loss = (sobel_magnitude * (output_image - hq) ** 2).sum()
                # loss = ((output_image - hq) ** 2).sum()
            elif loss_type == 'mse+dft':
                src = (output_image * 0.5 + 0.5) * 255.
                tgt = (hq * 0.5 + 0.5) * 255.

                src_luma_rounded, src_chroma_rounded = jpeg_encode(src.detach(), qf=qf)
                src_luma, src_chroma = jpeg_encode(src, qf=qf, rounded=False)
                tgt_luma_rounded, tgt_chroma_rounded = jpeg_encode(tgt, qf=100)

                unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
                fold = nn.Fold(output_size=(output_image.shape[-2], output_image.shape[-1]), kernel_size=(8, 8), stride=(8, 8))
                patches_sobel_magnitude = unfold(sobel_magnitude)
                patches_sobel_magnitude = patches_sobel_magnitude / (patches_sobel_magnitude.sum(dim=1, keepdim=True) + 1e-6)
                patches_sobel_magnitude = fold(patches_sobel_magnitude)

                # loss_luma = ((1 - patches_sobel_magnitude) * (src_luma + (src_luma_rounded - src_luma).detach() - tgt_luma_rounded) ** 2).mean()
                # loss_chroma = ((1 - patches_sobel_magnitude[:, :, ::2, ::2]) * (src_chroma + (src_chroma_rounded - src_chroma).detach() - tgt_chroma_rounded) ** 2).mean()
                loss_luma = ((src_luma + (src_luma_rounded - src_luma).detach() - tgt_luma_rounded) ** 2).mean()
                loss_chroma = ((src_chroma + (src_chroma_rounded - src_chroma).detach() - tgt_chroma_rounded) ** 2).mean()
                # loss_mse = ((1 - sobel_magnitude) * (output_image - hq) ** 2).sum()

                loss = loss_luma + loss_chroma

            # elif loss_type == 'vgg':
            #     vgg16 = models.vgg16(pretrained=True).features.to(output_image.device)
            #     vgg16.eval()
            #     x_vgg = vgg16(output_image)
            #     y_vgg = vgg16(hq)
            #     loss = nn.functional.mse_loss(x_vgg, y_vgg)
            # elif loss_type in ['dists', 'ssim']:
            #     loss = dists_metric(output_image * 0.5 + 0.5, hq * 0.5 + 0.5)
            else:
                raise NotImplementedError

            loss.backward()
            x_denoised = (x_denoised - x_denoised.grad * alpha).detach()

        with torch.no_grad():
            output_image = (
                self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
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


class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
            initialize_unet(args)

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                  device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):

        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
            ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            # breakpoint()
            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss