import os
import sys
import random
import argparse
import shutil
import numpy as np
from datetime import datetime

import torch
import timm
from utils.config import _C as cfg
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default='0')
parser.add_argument("--backbone", default=None)
parser.add_argument("--seed", default='0')

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
if args.gpuid is not None:
    cfg.gpuid = int(args.gpuid)
if args.backbone is not None:
    cfg.backbone = args.backbone
if args.seed is not None:
    cfg.seed = int(args.seed)

def set_seed():
    torch.cuda.set_device(cfg.gpuid)
    seed = cfg.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# Train
def train(epoch, dataloader, lambda_2):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0

    for batch_idx, (inputs, targets, _, index) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        total_loss = 0.0
        
        alpha = 0.20
        clean_mask = total_clean_idx[index].cuda()
        if clean_mask.sum() > 0:
            inputs_clean = inputs[clean_mask]
            targets_clean = targets[clean_mask]

            lam = np.random.beta(alpha, alpha)
            batch_size_clean = inputs_clean.size(0)
            index_perm = torch.randperm(batch_size_clean).cuda()

            mixed_inputs = lam * inputs_clean + (1 - lam) * inputs_clean[index_perm]
            targets_a = targets_clean
            targets_b = targets_clean[index_perm]

            outputs_clean = model(mixed_inputs)

            loss_clean = (
                lam * criterion(outputs_clean, targets_a) +
                (1 - lam) * criterion(outputs_clean, targets_b)
            ).mean()

            total_loss += loss_clean
        
        if cfg.dataset.startswith("cifar"):
            high_conf_mask = torch.zeros_like(clean_mask, dtype=torch.bool)
            soft_targets = []

            for idx_in_batch, sample_id in enumerate(index.tolist()):
                if sample_id in refined_samples_dict:
                    high_conf_mask[idx_in_batch] = True
                    soft_targets.append(refined_samples_dict[sample_id])
            
            if epoch >= 3:
                if high_conf_mask.sum() > 0:
                    soft_targets = torch.stack(soft_targets).cuda()
                    outputs_high = outputs[high_conf_mask]
                    
                    log_preds = F.log_softmax(outputs_high, dim=-1)
                    
                    tars_clamped = torch.clamp(soft_targets, min=1e-6) 
                    log_targets = torch.log(tars_clamped)
                    
                    preds = torch.exp(log_preds)
                    preds_clamped = torch.clamp(preds, min=1e-6)
                    
                    kl_pq = F.kl_div(log_preds, tars_clamped, reduction='batchmean')
                    kl_qp = F.kl_div(log_targets, preds_clamped, reduction='batchmean')

                    sym_kl = 0.5 * (kl_pq + kl_qp)
                    total_loss += lambda_2 * sym_kl      

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, total_loss.item()))
        sys.stdout.flush()

    return 100.*correct/total

def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

# ======== Data ========
if cfg.dataset.startswith("cifar"):
    from dataloader import dataloader_cifarN as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model)
    train_loader = loader.run('train')
    test_loader = loader.run('test')
elif cfg.dataset == "webvision":
    from dataloader import dataloader_webvision as dataloader
    train_loader, _, test_loader, _ = dataloader.build_loader(cfg)
elif cfg.dataset == "clothing1m":
    from dataloader import dataloader_clothing1M as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)

num_class = cfg.num_class

# ======== Model ========
if cfg.backbone == 'vit':
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/vit.npz'))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
elif cfg.backbone == 'resnet':
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/resnet.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
elif cfg.backbone == 'convnext':
    model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/convnext.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
elif cfg.backbone == 'mae':
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/mae.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
else:
    model, optimizer = load_clip(cfg)
    cfg.backbone == 'clip'

def load_samples(dataset: str, noise_mode: str, noise_ratio: float, base_dir: str = "./phase1_2"):
    folder = os.path.join(base_dir, dataset, str(cfg.seed))
    tag = f"{noise_mode}{noise_ratio}"
    refined_samples_path = os.path.join(folder, f"{tag}_refined_samples.pt")
    if not os.path.isfile(refined_samples_path):
        raise FileNotFoundError(f"Refined samples file not found: {refined_samples_path}")
    refined_samples_samples = torch.load(refined_samples_path)
    return refined_samples_samples

model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
total_clean_idx = torch.from_numpy(
    torch.load("./phase1_2/{}/{}/{}.pt".format(cfg.dataset, str(cfg.seed), cfg.noise_mode + str(cfg.noise_ratio)))
).bool()
refined_samples_samples = load_samples(cfg.dataset, cfg.noise_mode, cfg.noise_ratio, base_dir="./phase1_2")
refined_samples_dict = {sample_id: pseudo_label for sample_id, pseudo_label in refined_samples_samples}

best_acc = 0

sum1 = total_clean_idx.sum().item()
sum2 = len(refined_samples_samples)
lambda_2 = sum2 / (sum2 + sum1) - 0.1

for epoch in range(1, cfg.epochs + 1):
    train_acc = train(epoch, train_loader, lambda_2)
    test_acc = test(epoch, test_loader)
    best_acc = max(best_acc, test_acc)
    if epoch == cfg.epochs:
        print("Best Acc: %.2f Last Acc: %.2f" % (best_acc, test_acc))

    scheduler.step()
