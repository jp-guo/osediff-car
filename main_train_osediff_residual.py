import os
import os.path
import lpips
import math
import argparse
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import pyiqa
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from diffusion.osediff_residual import OSEDiff_reg, OSEDiff_gen

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from data.select_dataset import define_Dataset
from utils import utils_image as util
from utils import utils_option as option

from main_test_diff import get_validation_prompt
from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import warnings

warnings.filterwarnings("ignore")

import wandb
from datetime import datetime


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')


def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    # parser.add_argument("--revision", type=str, default=None, )
    # parser.add_argument("--variant", type=str, default=None, )
    # parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    # parser.add_argument("--output_dir", default='experience/osediff')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    # parser.add_argument("--train_batch_size", type=int, default=4,
    #                     help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000, ) # 100000
    parser.add_argument("--checkpointing_steps", type=int, default=5000, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--tracker_project_name", type=str, default="train_osediff",
                        help="The name of the wandb project to log to.")
    # parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=None,
    #                     help='A comma-separated list of integers')
    # parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1],
    #                     help='A comma-separated list of integers')
    # parser.add_argument("--deg_file_path", default=None, type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    parser.add_argument('--no_vsd', action='store_true')
    parser.add_argument('--test_epoch', type=int, default=1)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    parser.add_argument('--ram_ft_path', type=str, default=None)

    # dataset setting
    parser.add_argument("--datasets", default='options/train_diff_color.json')

    parser.add_argument('--prompt', type=str, default='', help='user prompts')

    parser.add_argument('--val_path', required=True)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    parser.add_argument('--train_decoder', action='store_true')

    parser.add_argument('--debug', action='store_true')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    args.tracker_project_name = os.path.join("training_results", args.tracker_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logging_dir = Path(args.tracker_project_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.tracker_project_name, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.tracker_project_name, "checkpoints"), exist_ok=True)
        if not args.debug:
            wandb.login(key="6b0e4eb09708be0cf3cb37658816b1000bcd16bb")
            wandb.init(project="diff-car", name=args.tracker_project_name)
        # os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = OSEDiff_gen(args)
    model_gen.set_train()
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # set vae adapter
    model_gen.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                      eps=args.adam_epsilon, )
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)

    # dataset_type = args.dataset_type
    args.datasets = option.parse_dataset(args.datasets)['datasets']
    for phase, dataset_opt in args.datasets.items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_set.normalize = True      # follow the setting of osediff
            # print('Dataset [{:s} - {:s}] is created.'.format(train_set.__class__.__name__, dataset_opt['name']))
            # train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            # logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            dl_train = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
            # args.max_train_steps = len(dl_train) * args.num_training_epochs
        # elif phase == 'test':
        #     test_set = define_Dataset(dataset_opt)
        #     test_set.normalize = True
        #     # print('Dataset [{:s} - {:s}] is created.'.format(test_set.__class__.__name__, dataset_opt['name']))
        #     dl_test = DataLoader(test_set, batch_size=1,
        #                              shuffle=False, num_workers=1,
        #                              drop_last=False, pin_memory=True)
        # else:
        #     raise NotImplementedError("Phase [%s] is not recognized." % phase)
    # breakpoint()
    # dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    # dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    # dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
    #                                        num_workers=args.dataloader_num_workers)
    # dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # init vlm model
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_vlm = ram(pretrained=args.ram_path,
                    pretrained_condition=None,
                    image_size=384,
                    vit='swin_l')
    model_vlm.eval()
    model_vlm.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        # args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        del args.datasets
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, total=args.max_train_steps)
    # breakpoint()
    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                x_src = batch["L"].to("cuda")
                x_tgt = batch["H"].to("cuda")

                qf_gt = batch['qf'].to("cuda").squeeze()
                B, C, H, W = x_src.shape
                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                batch["neg_prompt"] = [args.neg_prompt] * len(batch["prompt"])
                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = model_gen(x_src, batch=batch, args=args)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                # breakpoint()
                # KL loss
                loss_kl = 0
                if not args.no_vsd:
                    if torch.cuda.device_count() > 1:
                        loss_kl = model_reg.module.distribution_matching_loss(latents=latents_pred,
                                                                              prompt_embeds=prompt_embeds,
                                                                              neg_prompt_embeds=neg_prompt_embeds,
                                                                              args=args) * args.lambda_vsd
                    else:
                        loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds,
                                                                       neg_prompt_embeds=neg_prompt_embeds,
                                                                       args=args) * args.lambda_vsd
                loss = loss + loss_kl
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                diff loss: let lora model closed to generator
                """
                loss_d = 0
                if not args.no_vsd:
                    if torch.cuda.device_count() > 1:
                        loss_d = model_reg.module.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds,
                                                            args=args) * args.lambda_vsd_lora
                    else:
                        loss_d = model_reg.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds,
                                                     args=args) * args.lambda_vsd_lora
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item() if not args.no_vsd else 0
                    logs["loss_kl"] = loss_kl.detach().item() if not args.no_vsd else 0
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)
                    if not args.debug:
                        # wandb.log({'epoch': epoch, 'loss_l2': logs["loss_l2"], 'loss_lpips': logs['loss_lpips'],
                        #            'loss_d': logs["loss_d"], 'loss_kl': logs["loss_kl"]})
                        wandb.log({'epoch': epoch, 'loss_l2': logs["loss_l2"], 'loss_lpips': logs['loss_lpips']})
                        # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.tracker_project_name, "checkpoints", f"model_{global_step}.pkl")
                        # outf = os.path.join(args.tracker_project_name, "checkpoints", f"model.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                    accelerator.log(logs, step=global_step)

                    if global_step == args.max_train_steps:
                        accelerator.unwrap_model(model_gen).save_model(os.path.join(args.tracker_project_name, "checkpoints", f"model_max_train_steps.pkl"))

        if epoch % args.test_epoch == 0:
            lpips_metric = pyiqa.create_metric('lpips', device="cuda")
            dists_metric = pyiqa.create_metric('dists', device="cuda")
            niqe_metric = pyiqa.create_metric('niqe', device="cuda")
            musiq_metric = pyiqa.create_metric('musiq', device="cuda")
            maniqa_metric = pyiqa.create_metric('maniqa', device="cuda")
            clipiqa_metric = pyiqa.create_metric('clipiqa', device="cuda")

            if torch.cuda.device_count() > 1:
                model_gen.module.set_eval()
            else:
                model_gen.set_eval()

            H_paths = util.get_image_paths(args.val_path)

            quality_factor = 10
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnrb'] = []
            test_results['lpips'] = []
            test_results['dists'] = []
            test_results['niqe'] = []
            test_results['musiq'] = []
            test_results['maniqa'] = []
            test_results['clipiqa'] = []

            for idx, img in tqdm(enumerate(H_paths)):
                img_name, ext = os.path.splitext(os.path.basename(img))

                img_H = Image.open(img).convert('RGB')

                # vae can only process images with height and width multiples of 8
                new_width = img_H.width - img_H.width % 8
                new_height = img_H.height - img_H.height % 8
                img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

                img_L = img_H.copy()

                # img_L = utils.imread_uint(img, n_channels=n_channels)
                img_L = np.array(img_L)
                n_channels = img_L.shape[-1]
                if n_channels == 3:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
                if n_channels == 3:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

                img_L = Image.fromarray(img_L)
                # get caption
                # validation_prompt, lq = get_validation_prompt(args, img_L, DAPE)
                validation_prompt, lq = get_validation_prompt(args, img_L, model_vlm)
                # translate the image
                with torch.no_grad():
                    lq = lq * 2 - 1
                    if torch.cuda.device_count() > 1:
                        img_E = model_gen.module.eval(lq, prompt=validation_prompt)
                    else:
                        img_E = model_gen.eval(lq, prompt=validation_prompt)
                    img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
                    if args.align_method == 'adain':
                        img_E = adain_color_fix(target=img_E, source=img_L)
                    elif args.align_method == 'wavelet':
                        img_E = wavelet_color_fix(target=img_E, source=img_L)
                    else:
                        pass
                img_H = np.array(img_H)
                img_E = np.array(img_E)

                psnr = util.calculate_psnr(img_E, img_H, border=0)
                ssim = util.calculate_ssim(img_E, img_H, border=0)
                psnrb = util.calculate_psnrb(img_H, img_E, border=0)

                util.imsave(img_E, os.path.join(args.tracker_project_name, img_name + '.png'))

                img_E, img_H = img_E / 255., img_H / 255.
                img_E, img_H = torch.tensor(img_E, device="cuda").permute(2, 0, 1).unsqueeze(0), torch.tensor(img_H,
                                                                                                              device="cuda").permute(
                    2, 0, 1).unsqueeze(0)
                img_E, img_H = img_E.type(torch.float32), img_H.type(torch.float32)

                lpips_score = lpips_metric(img_E, img_H)
                dists = dists_metric(img_E, img_H)
                niqe = niqe_metric(img_E, img_H)
                musiq = musiq_metric(img_E, img_H)
                maniqa = maniqa_metric(img_E, img_H)
                clipiqa = clipiqa_metric(img_E, img_H)

                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['psnrb'].append(psnrb)
                test_results['lpips'].append(lpips_score.item())
                test_results['dists'].append(dists.item())
                test_results['niqe'].append(niqe.item())
                test_results['musiq'].append(musiq.item())
                test_results['maniqa'].append(maniqa.item())
                test_results['clipiqa'].append(clipiqa.item())

            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
            avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            avg_dists = sum(test_results['dists']) / len(test_results['dists'])
            avg_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
            avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
            avg_maniqa = sum(test_results['maniqa']) / len(test_results['maniqa'])
            avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])
            # print(
            #     'Average PSNR/SSIM/PSNRB - {} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'.format(
            #         str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))
            if accelerator.is_main_process and not args.debug:
                wandb.log(
                    {'PSNR': ave_psnr, 'LPIPS': avg_lpips, 'DISTS': avg_dists, 'NIQE': avg_niqe, 'MANIQA': avg_maniqa})

            if torch.cuda.device_count() > 1:
                model_gen.module.set_train()
            else:
                model_gen.set_train()

if __name__ == "__main__":
    # from diffusers import DiffusionPipeline
    #
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

    args = parse_args()
    main(args)
