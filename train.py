import argparse
import logging
import random
import shutil
import sys
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from medpy import metric
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from transformers import AutoTokenizer, AutoModel

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from monai.losses import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss

from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    ValGenerator,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import losses, metrics, ramps, val_2d
from evaluation_metrics import calculate_metric_percase

import ml_collections

# Model setting.



from losses.spatial_prior_loss import SpatialPriorGeometryLoss
from networks.span_net import SPANNet
net_name = "SPAN-Net"

def build_model(num_classes):
    return SPANNet(n_channels=3, n_classes=num_classes, text_dim=768)


# Main program.



parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True, help="Numbers of Experiment")
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=[
        "BKAI", "BRISC", "BUID", "BUSBRA", "BUSUC", "ClinicDB", "ColonDB",
        "CVC300", "ISIC", "Kvasir", "UDIAT", "UWaterlooSkinCancer",
        "BUSI", "Covid19", "BTMRI"
    ],
    help="dataset name for selecting text generation rule",
)
parser.add_argument("--max_epoch", type=int, default=150, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.0001, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[224, 224], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--early_stopping_patience", type=int, default=50, help="early stopping patience (epochs)")
parser.add_argument("--text_ratio", type=float, default=0.5, help="ratio of samples where text modality is enabled")
args = parser.parse_args()

args.root_path = f"../data/{args.dataset_name}"
args.model = f"{net_name}_{args.exp}/{args.dataset_name}"

def read_text(filename):
    df = pd.read_excel(filename)
    return df.to_dict(orient="records")

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

def set_deterministic(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def test_single_batch(image, label, text, net, classes):
    if isinstance(label, torch.Tensor):
        label = label.squeeze(0).cpu().numpy()
    image = image.cuda().float()
    text = text.cuda().float()

    net.eval()
    with torch.no_grad():
        out_main = net(image, text)[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()

    per_class_metrics = []
    for cls in range(1, classes):
        if np.sum(label == cls) == 0:
            per_class_metrics.append((np.nan,) * 8)
        else:
            metrics = calculate_metric_percase(pred == cls, label == cls)
            per_class_metrics.append(metrics)

    return np.array(per_class_metrics, dtype=np.float64)


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 2
    return config


def train(args, snapshot_path):
    set_deterministic(args.seed)

    base_lr = args.base_lr
    num_classes = args.num_classes
    max_epoch = args.max_epoch

    model = build_model(num_classes)
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    text_embedder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Set dataset paths.
    train_img_path = os.path.join(args.root_path, "Train_Folder")
    val_img_path = os.path.join(args.root_path, "Val_Folder")
    prompt_path = os.path.join(args.root_path, "Prompts_Folder")
    
    # Load text prompts.
    train_text = read_text(os.path.join(prompt_path, "Train_text.xlsx"))
    val_text = read_text(os.path.join(prompt_path, "Val_text.xlsx"))

    # Build datasets.
    db_train = BaseDataSets(
        dataset_path=train_img_path,
        row_text=train_text,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        tokenizer=tokenizer,
        text_embedder=text_embedder
    )
    
    db_val = BaseDataSets(
        dataset_path=val_img_path,
        row_text=val_text,
        transform=ValGenerator(args.patch_size),
        tokenizer=tokenizer,
        text_embedder=text_embedder
    )

    generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        shuffle=True,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=1e-5,
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
    geo_loss = SpatialPriorGeometryLoss()

    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    best_performance = 0.0
    early_stopping_counter = 0
    patience = args.early_stopping_patience
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, text = (
                sampled_batch["image"],
                sampled_batch["label"],
                sampled_batch["text"],
            )
            volume_batch, label_batch, text = (
                volume_batch.cuda(),
                label_batch.cuda(),
                text.cuda(),
            )

            outputs, priors_list, _, _ = model(volume_batch, text)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch.unsqueeze(1))
            loss_geo = geo_loss(priors_list, label_batch)
            loss = 0.5 * loss_dice + 0.5 * loss_ce + 0.4 * loss_geo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            logging.info(
                "iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %.6f, loss_geo: %f"
                % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_, loss_geo.item())
            )

        if iter_num > 0 and iter_num % len(trainloader) == 0:
            model.eval()
            
            # Collect validation metrics.
            all_metrics = []

            for _, sampled_batch in enumerate(valloader):
                metric_i = test_single_batch(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    sampled_batch["text"],
                    model,
                    classes=num_classes,
                )
                all_metrics.append(metric_i)


            all_metrics = np.array(all_metrics)

            # Average over validation samples.
            class_metrics = np.nanmean(all_metrics, axis=0)

            # Metric columns: Dice at 0, HD95 at 2.


            performance = np.nanmean(class_metrics[:, 0])
            mean_hd95 = np.nanmean(class_metrics[:, 2])

            if performance > best_performance:
                best_performance = performance
                early_stopping_counter = 0

                save_best_path = os.path.join(
    snapshot_path, "best_model.pth"
						)
                torch.save(model.state_dict(), save_best_path)

                logging.info(
                    f"Iteration {iter_num}: Dice improved to {performance:.4f}, saving model."
                )
            else:
                early_stopping_counter += 1
                logging.info(
                    f"Iteration {iter_num}: No improvement. "
                    f"EarlyStopping {early_stopping_counter}/{patience}"
                )

            logging.info(
                "iteration %d : mean_dice : %f mean_hd95 : %f"
                % (iter_num, performance, mean_hd95)
            )

            model.train()

            if early_stopping_counter >= patience:
                logging.info("Early stopping triggered.")
                iterator.close()
                break
            if iter_num >= max_iterations:
                logging.info("Max iterations reached.")
                break
    logging.info("Training finished.")
    return model


if __name__ == "__main__":
    snapshot_path = f"../model/{args.model}"
    if os.path.exists(snapshot_path):
        raise RuntimeError("Snapshot folder already exists")
    os.makedirs(snapshot_path, exist_ok=True)

    model_code_path = os.path.join(snapshot_path, args.model)
    src_model_path = os.getcwd()

    if os.path.exists(model_code_path):
        shutil.rmtree(model_code_path)

    shutil.copytree(
        src_model_path,
        model_code_path,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))
    train(args, snapshot_path)
