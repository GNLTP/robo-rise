import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
from datetime import datetime

from tqdm import tqdm
from policy import RISE
from copy import deepcopy
import torch.nn.functional as F
from utils.utils import *
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.training import set_seed, plot_history, sync_loss
from dataset.data_factory import data_provider

@torch.inference_mode()
def evaluate(model, val_loader, device, open_loop: bool = False):
    model.eval()
    total, n = 0.0, 0

    pbar = tqdm(
        val_loader, total=len(val_loader), desc="Validating",
        dynamic_ncols=True, leave=False, mininterval=0.1
    )

    for batch in pbar:
        cloud_coords = batch["input_coords_list"].to(device)
        cloud_feats = batch["input_feats_list"].to(device).contiguous()
        cloud_data = ME.SparseTensor(features=cloud_feats, coordinates=cloud_coords, device=device)
        action = batch["action"].to(device).float()
        B = action.shape[0]

        if open_loop == False:
            loss_tf = model(cloud_data, action, batch_size=B)
            loss_val = float(loss_tf.detach().item())
        else:
            pred = model(cloud_data, actions=None, batch_size=B)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.to(dtype=action.dtype)
            if pred.shape != action.shape:
                raise ValueError(
                    f"Shape mismatch: pred_action {tuple(pred.shape)} != action {tuple(action.shape)}"
                )

            loss_mse = F.mse_loss(pred, action)
            loss_val = float(loss_mse.detach().item())

        total += loss_val
        n += 1
        pbar.set_postfix(val_loss=total / max(1, n))

    return total / max(1, n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--path_time', type=str, default='2025-08-24_09-58-38')
    parser.add_argument('--target_obj', type=str, default='banana')
    parser.add_argument('--cam_ids', type=str, default='agentview')
    parser.add_argument('--num_obs', type=int, default=1)
    parser.add_argument('--num_action', type=int, default=20)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--with_cloud', type=bool, default=False)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--aug_trans_min', type=float, nargs=3, default=[-0.2, -0.2, -0.2])
    parser.add_argument('--aug_trans_max', type=float, nargs=3, default=[0.2, 0.2, 0.2])
    parser.add_argument('--aug_rot_min', type=float, nargs=3, default=[-30, -30, -30])
    parser.add_argument('--aug_rot_max', type=float, nargs=3, default=[30, 30, 30])
    parser.add_argument('--aug_jitter', type=bool, default=False)
    parser.add_argument('--aug_jitter_params', type=float, nargs=4, default=[0.4, 0.4, 0.2, 0.1])
    parser.add_argument('--aug_jitter_prob', type=float, default=0.2)
    parser.add_argument('--obs_feature_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints")
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    set_seed(args.seed)
    if use_cuda:
        torch.cuda.set_device(0)

    print("Loading dataset ...")
    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")

    print("Loading policy ...")
    model = RISE(
        num_action=args.num_action,
        input_dim=6,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    ).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    if args.resume_ckpt is not None:
        model.load_state_dict(torch.load(args.resume_ckpt, map_location=device), strict=False)
        print(f"Checkpoint {args.resume_ckpt} loaded.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = f"{args.data_path}/{args.ckpt_dir}/{timestamp}/{args.target_obj}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)  # 优化器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=len(train_loader) * args.num_epochs,
    )
    lr_scheduler.last_epoch = len(train_loader) * (args.resume_epoch + 1) - 1 # 选取使用第几个epoch开始训练

    best_val = float('inf')
    best_path = os.path.join(ckpt_dir, f"best_val_seed_{args.seed}.ckpt")
    hist_train, hist_val, save_list = [], [], []

    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        total, n = 0.0, 0

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.1,
        )

        for batch in pbar:
            optimizer.zero_grad()

            cloud_coords = batch["input_coords_list"].to(device)
            cloud_feats = batch["input_feats_list"].to(device).contiguous()
            cloud_data = ME.SparseTensor(features=cloud_feats, coordinates=cloud_coords, device=device)
            action = batch["action"].to(device)
            action_nor = batch["action_normalized"].to(device)

            loss = model(cloud_data, action, batch_size=action.shape[0])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total += float(loss.item());
            n += 1
            pbar.set_postfix(train_loss=total / max(1, n), lr=optimizer.param_groups[0]["lr"])

        avg_train = total / max(1, n)
        hist_train.append(avg_train)
        print(f"Train loss: {avg_train:.6f}")

        val_loss = evaluate(model, val_loader, device, open_loop=False)
        hist_val.append(val_loss)
        print(f"Val loss: {val_loss:.6f}")

        saved_this_epoch = False

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"New best (val_loss={best_val:.6f}). Saved to: {best_path}")
            saved_this_epoch = True

        save_list.append(1 if saved_this_epoch else 0)

        if (epoch + 1) % args.save_epochs == 0:
            snap_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{args.seed}.ckpt")
            torch.save(model.state_dict(), snap_path)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "policy_last.ckpt"))
    plot_loss_curves_with_dual_style(hist_train, hist_val, save_list, ckpt_dir)

