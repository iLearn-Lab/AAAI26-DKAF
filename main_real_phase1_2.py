import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from model import *
from collections import defaultdict
from utils.config import _C as cfg
from utils.lnl_methods import *

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", choices=['sym', 'idn'], default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default='0')
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

def train(epoch, dataloader, M, vis_proto):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0
    precision, recall = 0, 0
    
    p_logits_dict = defaultdict(list)
    high_conf_list = []
    
    cur_tau = 0.95
    global mix_alpha

    text_correct_num = 0
    vision_correct_num = 0

    avg_loss = 0
    
    for batch_idx, (inputs, targets, comple_targets, index) in enumerate(dataloader):
        inputs, targets, comple_targets = inputs.cuda(), targets.cuda(), comple_targets.cuda()
        pos_outputs, neg_outputs, neg_text_features, image_features = model(inputs, return_neg=True)
        
        probs_text = F.softmax(pos_outputs.detach(), dim=-1)
        maxp, pseudo = probs_text.max(dim=-1)
        
        if epoch > 1:
            feats_vision = F.normalize(image_features, dim=-1, eps=1e-6)
            vis_proto_vision = F.normalize(vis_proto, dim=-1, eps=1e-6)
            logits_vision = model.logit_scale.exp() * feats_vision.detach() @ vis_proto_vision.detach().T
            
            probs_vision = F.softmax(logits_vision.detach(), dim=-1)
            maxp_vision, pseudo_vision = probs_vision.max(dim=-1)
            
            probs_mix = mix_alpha * probs_text + (1 - mix_alpha) * probs_vision
            maxp_mix, pseudo_mix = probs_mix.max(dim=-1)

        total_loss = 0
        
        p_clean_list = []
        is_max_list = []
        loss_no = 0
        loss_yes = 0
        for i in range(neg_outputs.shape[0]):
            logits_per_image_i = torch.cat([pos_outputs.unsqueeze(-1), neg_outputs[i].unsqueeze(-1)], dim=-1)
            probs_per_image_i = F.softmax(logits_per_image_i.detach(), dim=-1)
            p_yes_i = probs_per_image_i[:, :, 0].squeeze(-1)
            
            _, neg_predicted = p_yes_i.max(1)
            is_max_i = neg_predicted == targets
            is_max_list.append(is_max_i.float())
            
            p_clean_i = p_yes_i[range(inputs.shape[0]), targets]
            p_clean_list.append(p_clean_i)
            
            logits_log_softmax_i = F.log_softmax(logits_per_image_i, dim=-1)
            p_yes_log_probs = logits_log_softmax_i[:, :, 0].squeeze(-1)
            p_no_log_probs = logits_log_softmax_i[:, :, 1].squeeze(-1)
            loss_list = []
            for i in range(comple_targets.shape[1]):
                neg_target_i = comple_targets[:, i]
                loss_i = neg_criterion(p_no_log_probs, neg_target_i)
                loss_list.append(loss_i)
            loss_no_i = torch.stack(loss_list, dim=1).mean(dim=1).mean()
            loss_no += loss_no_i
            
            with torch.no_grad():
                if epoch <= 3:
                    idx_select_i = ((p_clean_i > 0.5) & is_max_i).cpu()
                else:
                    momentum_ema = 0.7
                    p_clean_i = momentum_ema * p_clean_ema[index] + (1 - momentum_ema) * p_clean_i
                    idx_select_i = ((p_clean_i > 0.5) & is_max_i).cpu()
            
            if epoch <= 1:
                loss_yes_i = neg_criterion(p_yes_log_probs, targets).mean()
            else:
                pseudo_labels = targets.clone()
                pseudo_labels[~idx_select_i] = pseudo_mix[~idx_select_i]
                loss_yes_i = neg_criterion(p_yes_log_probs, pseudo_labels).mean()
            loss_yes += loss_yes_i
        total_loss += (loss_no / neg_outputs.shape[0])
        total_loss += (loss_yes / neg_outputs.shape[0])
            
        p_clean = torch.stack(p_clean_list, dim=0).mean(dim=0)
        is_max = torch.stack(is_max_list, dim=0).mean(dim=0) > 0.5 
        with torch.no_grad():
            if epoch <= 3:
                p_clean_ema[index] = p_clean
            else:
                momentum_ema = 0.7
                p_clean_ema[index] = momentum_ema * p_clean_ema[index] + (1 - momentum_ema) * p_clean
        idx_select = ((p_clean_ema[index] > 0.5) & is_max).cpu()
        total_clean_idx[index] = idx_select
        
        for i, sid in enumerate(index):
            if(idx_select[i]):
                total_pseudos[sid.item()] = targets[i].item()
                high_conf_list.append((sid.item(), targets[i].item()))
                if epoch > 2: 
                    if(targets[i].item() == pseudo[i].item()):
                        text_correct_num += 1
                    if(targets[i].item() == pseudo_vision[i].item()):
                        vision_correct_num += 1
            else:
                if(epoch <= 1):
                    if(maxp[i].item() >= cur_tau):
                        total_pseudos[sid.item()] = pseudo[i].item()
                        high_conf_list.append((sid.item(), pseudo[i].item()))
                else:
                    if(maxp_mix[i].item() >= cur_tau):
                        total_pseudos[sid.item()] = pseudo_mix[i].item()
                        high_conf_list.append((sid.item(), pseudo_mix[i].item()))
        
        if epoch == cfg.epochs:
            for i in range(inputs.shape[0]):
                if(idx_select[i]):
                    class_id = targets[i].item()
                    p_logits_dict[class_id].append(maxp_mix[i].item())

        loss_cls = 0 
        if epoch <= cfg.warmup:
            per_sample_loss = robust_criterion(pos_outputs, targets)
            loss_cls = per_sample_loss.mean()
        else: 
            clean_logits = pos_outputs[idx_select]
            clean_labels = targets[idx_select]
            B, C = clean_logits.shape
            margin_rows = M[clean_labels]
            row_idx = torch.arange(B, device=clean_labels.device)
            mask = torch.zeros_like(margin_rows, dtype=torch.bool)
            mask[row_idx, clean_labels] = True
            adjusted = clean_logits + margin_rows * (~mask)
            loss_cls = F.cross_entropy(adjusted, clean_labels, reduction='mean')   
        total_loss += loss_cls 
        
        diff_loss = 0
        num_cls = pos_outputs.shape[1]
        for c in range(num_cls):
            neg_feats_c = neg_text_features[:, c, :]
            neg_feats_c = F.normalize(neg_feats_c, dim=-1)
            
            sim_matrix = neg_feats_c @ neg_feats_c.T
            triu_indices = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
            sim_values = sim_matrix[triu_indices[0], triu_indices[1]]
            avg_sim = sim_values.mean() if sim_values.numel() > 0 else 0.
            
            diff_loss += avg_sim
        diff_loss = diff_loss / num_cls
        total_loss += (0.5 * diff_loss)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += targets.size(0)

        avg_loss += total_loss

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] total_loss: %.4f\t'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, total_loss))       
        sys.stdout.flush()
    
    avg_loss = avg_loss/num_iter
    print("\n| Train Epoch #%d\t avg_loss: %.2f\t" % (epoch, avg_loss))

    if epoch > 2:
        mix_alpha = text_correct_num / (text_correct_num + vision_correct_num + 1e-6)

    if epoch == cfg.epochs:
        return None, None, None, high_conf_list, p_logits_dict
    else:
        return None, None, None, high_conf_list, None

def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, return_sim=True)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            total_preds.append(predicted)
            total_targets.append(targets)

    acc = 100. * correct / total
    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)

    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

if cfg.dataset.startswith("cifar"):
    from dataloader import dataloader_cifarN as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model)
    train_loader = loader.run('train')
    test_loader = loader.run('test')
elif cfg.dataset == "clothing1m":
    from dataloader import dataloader_clothing1M as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "webvision":
    from dataloader import dataloader_webvision as dataloader
    train_loader, _, test_loader, _ = dataloader.build_loader(cfg)

num_class = cfg.num_class

model, optimizer = load_deft(cfg)
model.cuda()
neg_criterion = torch.nn.NLLLoss(reduction='none')
robust_criterion = SCELoss(alpha=1, beta=1, num_classes=cfg.num_class, reduction="none")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

@torch.no_grad()
def compute_global_similarity_on_clean(model, loader, total_pseudos):
    device = next(model.parameters()).device
    total_pseudos = total_pseudos.to(device)
    C = cfg.num_class
    D = model.image_encoder.proj.shape[0]
    sums = torch.zeros(C, D, device=device)
    counts = torch.zeros(C, device=device)
    model.eval()
    for images, labels, _, indices in loader:
        images, labels = images.cuda(), labels.cuda()
        mask = total_clean_idx[indices].cuda()
        if mask.sum() == 0:
            continue
        feats = model.image_encoder(images, model.tuner)
        pseudos_feats = feats[mask]
        pseudos_lables = labels[mask]
        for f, c in zip(pseudos_feats, pseudos_lables):
            sums[c] += f
            counts[c] += 1
    counts = counts.clamp(min=1.0)
    vis_proto = sums / counts.unsqueeze(1)
    vis_proto = vis_proto.to(dtype=model.image_encoder.dtype)
    prompts = model.prompt_learner()
    tok     = model.tokenized_prompts
    txt_proto, _ = model.text_encoder(prompts, tok)
    V = F.normalize(vis_proto, dim=1, eps=1e-6)
    T = F.normalize(txt_proto, dim=1, eps=1e-6)
    S_vis = V @ V.t()
    S_txt = T @ T.t()
    S = torch.max(S_vis, S_txt)
    return S, vis_proto

@torch.no_grad()
def compute_margin_matrix(S, sigma, m_scale=9):
    device = S.device
    sigma = sigma.to(device)
    max_sigma = sigma.max().clamp(min=1.0)
    delta = 1 - (sigma / max_sigma)
    m_vec = m_scale * delta
    M = S * m_vec.view(-1, 1)
    return M

M = None
vis_proto = None
mix_alpha = 0.5
sigma = torch.zeros(cfg.num_class).cuda()
total_clean_idx = torch.zeros(len(train_loader.dataset), dtype=torch.bool)
total_pseudos = torch.zeros(len(train_loader.dataset), dtype=torch.long)
p_clean_ema = torch.zeros(len(train_loader.dataset), dtype=torch.float16).cuda()

for epoch in range(1, cfg.epochs + 1):
    total_clean_idx[:] = False
    total_pseudos[:] = -1
    
    train_acc, precision, recall, high_conf_list, p_logits_list = train(epoch, train_loader, M, vis_proto)
    scheduler.step()
    
    if epoch >= 1:
        sigma.zero_()
        for sample_id, label in high_conf_list:
            sigma[label] += 1 
        S, vis_proto = compute_global_similarity_on_clean(model, train_loader, total_pseudos)
        M = compute_margin_matrix(S, sigma, m_scale=9)
    
    if epoch == cfg.epochs:
        threshold_dict = {}
        tau_c = 0.2
        beta = 0.2 
        max_freq = 0
        min_threshold = 0.05
        for c in range(cfg.num_class):
            max_freq = max(max_freq, len(p_logits_list[c]))
        for c in range(cfg.num_class):
            c_freq = len(p_logits_list[c])
            if c_freq == 0:
                threshold_dict[c] = min_threshold
            else:
                tau_c_adjusted = tau_c - beta * (1 - c_freq / (max_freq + 1e-5))
                threshold_dict[c] = max(tau_c_adjusted, min_threshold)
        
        refined_samples = []
        model.eval()

        des = os.path.join("phase1", cfg.dataset, str(cfg.seed))
        os.makedirs(des, exist_ok=True)
        if cfg.dataset.startswith("cifar"):
            print("Phase 2 begin...")
            with torch.no_grad():
                for batch_idx, (inputs, targets, comple_targets, index) in enumerate(train_loader):
                    inputs = inputs.cuda()
                    
                    logits_text, _, _, feats = model(inputs, return_neg=True)
                    probs_text = F.softmax(logits_text.detach(), dim=-1)
                    
                    feats_vision = F.normalize(feats, dim=-1, eps=1e-6)
                    vis_proto_vision = F.normalize(vis_proto, dim=-1, eps=1e-6)
                    logits_vision = model.logit_scale.exp() * feats_vision.detach() @ vis_proto_vision.detach().T
                    probs_vision = F.softmax(logits_vision.detach(), dim=-1)
                    
                    probs_mix = mix_alpha * probs_text + (1 - mix_alpha) * probs_vision
                    maxp_mix, pseudo_mix = probs_mix.max(dim=-1)
                    
                    for i in range(inputs.shape[0]):
                        sample_id = index[i].item()  
                        
                        if total_clean_idx[sample_id]:
                            continue

                        pred_class_mix = pseudo_mix[i].item()
                        
                        if mix_alpha > 0.5:
                            if probs_text[i].argmax() != pred_class_mix: continue
                        else:
                            if probs_vision[i].argmax() != pred_class_mix: continue
                            
                        delta_mix = (probs_mix[i].max() - torch.topk(probs_mix[i], 2).values[1]).item()
                        if delta_mix <= threshold_dict[pred_class_mix]:
                            continue
                        
                        refined_samples.append((sample_id, probs_mix[i].detach().cpu()))
            
            prefix = f"{cfg.noise_mode}{cfg.noise_ratio}"
            torch.save(refined_samples, os.path.join(des, f"{prefix}_refined_samples.pt"))
        torch.save(total_clean_idx.numpy(), os.path.join(des, f"{prefix}.pt"))
        if cfg.dataset.startswith("cifar"):
            print("Phase 2 end...")
        else:
            print("Phase 2 skipped...")
