import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected, to_undirected
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score, accuracy_score
from rdkit import Chem
import copy
import torch_geometric.data.batch as DataBatch
from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, \
    set_seed, process_data, relabel
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, \
    init_metric_dict
from sklearn.metrics import roc_auc_score, accuracy_score


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class,
                 multi_label, random_state,
                 method_config, shared_config, model_config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        self.model_name = model_config['model_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.cur_pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']
        self.cur_info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)
        self.sel_r = method_config.get('sel_r', 0.5)

        self.from_scratch = method_config['from_scratch']
        self.save_mcmc = method_config.get('save_mcmc', False)
        self.from_mcmc = method_config.get('from_mcmc', False)
        self.multi_linear = method_config.get('multi_linear', False)
        self.mcmc_dir = method_config['mcmc_dir']
        self.pre_model_name = method_config['pre_model_name']

        if self.multi_linear in [5552]:
            self.fc_proj = nn.Sequential(nn.Sequential(nn.Dropout(p=0.33),
                                                       nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                                                       nn.BatchNorm1d(self.clf.hidden_size),
                                                       nn.ReLU(inplace=True),
                                                       nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                                                       ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(
                list(extractor.parameters()) + list(clf.parameters()) + list(self.fc_proj.parameters()), lr=lr,
                weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max',
                                                                                   **scheduler_config)
        if self.multi_linear in [5553, 5554]:
            class_dim = 1 if num_class == 2 and not multi_label else num_class
            self.fc_proj = nn.Sequential(
                nn.Sequential(nn.Linear(self.clf.hidden_size, class_dim),
                              nn.BatchNorm1d(class_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(class_dim, class_dim),
                              ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(
                list(extractor.parameters()) + list(clf.parameters()) + list(self.fc_proj.parameters()), lr=lr,
                weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max',
                                                                                   **scheduler_config)

        if self.multi_linear in [5550, 5552, 5553, 5554, 5555, 5559, 5449, 5229, 5669]:
            if not self.from_mcmc:
                self.fc_out = self.clf
            self.fc_out = get_model(model_config['x_dim'], model_config['edge_attr_dim'], num_class,
                                    model_config['multi_label'], model_config, device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            if self.multi_linear in [5552, 5554, 5555, 5669, 5449]:
                self.fc_out.load_state_dict(copy.deepcopy(self.clf.state_dict()))
            self.optimizer = torch.optim.Adam(self.fc_out.parameters(), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max',
                                                                                   **scheduler_config)

        self.sampling_trials = method_config.get('sampling_trials', 100)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)

    def __loss__(self, att, clf_logits, clf_labels, epoch, training=False, agg='mean'):
        if clf_logits.size(0) != clf_labels.size(0):
            pred_losses = []
            for i in range(clf_logits.size(0)):
                pred_losses.append(self.criterion(clf_logits[i, :], clf_labels))
            if agg.lower() == 'max':
                pred_loss = torch.stack(pred_losses).max()
            else:
                pred_loss = torch.stack(pred_losses).mean()
        else:
            pred_losses = None
            pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.final_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                       init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()
        pred_lossc = pred_loss * self.pred_loss_coef
        info_lossc = info_loss * self.cur_info_loss_coef
        loss = pred_lossc + info_lossc
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        if pred_losses != None:
            for i, pl in enumerate(pred_losses):
                loss_dict[f'pred_L{i}'] = pl.item()
        if training:
            self.optimizer.zero_grad()
            pred_lossc.backward(retain_graph=True)
            pred_grad = []
            for param in self.extractor.parameters():
                if param.grad != None:
                    pred_grad.append(param.grad.data.clone().flatten().detach())
            pred_grad = torch.cat(pred_grad) if len(pred_grad) > 0 else torch.zeros([1]).to(loss.device)
            self.optimizer.zero_grad()
            info_lossc.backward(retain_graph=True)
            info_grad = []
            for param in self.extractor.parameters():
                if param.grad != None:
                    info_grad.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            info_grad = torch.cat(info_grad) if len(pred_grad) > 0 else torch.zeros([1])
            grad_sim = F.cosine_similarity(pred_grad.unsqueeze(0), info_grad.unsqueeze(0)).to(loss.device)
            loss_dict['grad_sim'] = grad_sim.item()
            loss_dict['pred_grad'] = pred_grad.norm().item()
            loss_dict['info_grad'] = info_grad.norm().item()
        return loss, loss_dict

    def attend(self, data, att_log_logits, epoch, training):
        att = self.sampling(att_log_logits, epoch, training)
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
        return edge_att

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)

        edge_att = self.attend(data, att_log_logits, epoch, training)
        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch, training)

        edge_att = att_log_logits.sigmoid().detach()
        if self.learn_edge_att and is_undirected(data.edge_index):
            trans_idx, trans_val = transpose(data.edge_index, edge_att, None, None, coalesced=False)
            trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
            edge_att = (edge_att + trans_val_perm) / 2
        elif not self.learn_edge_att:
            edge_att = self.lift_node_att_to_edge_att(edge_att, data.edge_index)

        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        self.eval()
        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        self.train()
        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase_str = 'test ' if phase == 'test' else phase

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_original_clf_logits = [], [], [], [], []

        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)
            has_edges_mask = torch.tensor([g.edge_index.numel() > 0 for g in data.to_data_list()], dtype=torch.bool)
            with torch.no_grad():
                device_data = data.to(self.device)
                original_clf_logits = self.clf(device_data.x, device_data.edge_index, device_data.batch,
                                               edge_attr=device_data.edge_attr)

            desc, _, _, _, _, _ = self.log_epoch(epoch, phase_str, loss_dict, data.edge_label.data.cpu(), att,
                                                 data.y.data.cpu(), clf_logits, original_clf_logits.data.cpu(),
                                                 batch=True,
                                                 has_edges_mask=has_edges_mask)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(data.edge_label.data.cpu())
            all_att.append(att)
            all_clf_labels.append(data.y.data.cpu())
            all_clf_logits.append(clf_logits)
            all_original_clf_logits.append(original_clf_logits.data.cpu())

            if idx == loader_len - 1:
                all_exp_labels = torch.cat(all_exp_labels)
                all_att = torch.cat(all_att)
                all_clf_labels = torch.cat(all_clf_labels)
                all_clf_logits = torch.cat(all_clf_logits)
                all_original_clf_logits = torch.cat(all_original_clf_logits)
                all_has_edges_mask = torch.cat(
                    [torch.tensor([g.edge_index.numel() > 0 for g in batch.to_data_list()], dtype=torch.bool) for batch
                     in data_loader])

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len

                desc, explanation_accuracy, distillation_accuracy, fidelity, avg_loss, auc = self.log_epoch(
                    epoch, phase_str, all_loss_dict, all_exp_labels, all_att,
                    all_clf_labels, all_clf_logits, all_original_clf_logits, batch=False,
                    has_edges_mask=all_has_edges_mask)
            pbar.set_description(desc)

        return explanation_accuracy, distillation_accuracy, fidelity, avg_loss, auc

    def train_self(self, loaders, test_set, metric_dict, use_edge_attr):
        distillation_start_time = time.time()

        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'val', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)

            if self.scheduler is not None:
                self.scheduler.step(valid_res[1])

            if (valid_res[1] > metric_dict['metric/distillation_accuracy'] or
                    (valid_res[1] == metric_dict['metric/distillation_accuracy'] and valid_res[3] < metric_dict[
                        'metric/best_val_loss'])):

                distillation_time = time.time() - distillation_start_time
                if 'tree' in self.dataset_name.lower():
                    final_metrics_res = train_res
                else:
                    final_metrics_res = test_res
                metric_dict.update({
                    'metric/best_epoch': epoch,
                    'metric/best_val_loss': valid_res[3],
                    'metric/explanation_accuracy': final_metrics_res[0],
                    'metric/distillation_time': distillation_time,
                    'metric/distillation_accuracy': final_metrics_res[1],
                    'metric/fidelity': final_metrics_res[2],
                    'metric/AUC': final_metrics_res[4]
                })
                if self.save_mcmc:
                    save_checkpoint(self.clf, self.mcmc_dir, model_name=self.pre_model_name + f"_clf_mcmc")
                    save_checkpoint(self.extractor, self.mcmc_dir, model_name=self.pre_model_name + f"_att_mcmc")

            for metric, value in metric_dict.items():
                self.writer.add_scalar(f'best/{metric.split("/")[-1]}', value, epoch)

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_epoch"]}, '
                  f'Test Distill Acc: {metric_dict["metric/distillation_accuracy"]:.4f}, '
                  f'Test Explan Acc: {metric_dict["metric/explanation_accuracy"]:.4f}, '
                  f'Test Fidelity: {metric_dict["metric/fidelity"]:.4f},'
                  f' Test AUC: {metric_dict["metric/AUC"]:.4f}')
            print('=' * 80)

        _ = self.run_one_epoch(loaders['test'], self.epochs, 'test', use_edge_attr)
        total_runtime = time.time() - distillation_start_time
        metric_dict['metric/overall_runtime'] = total_runtime

        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, clf_labels, clf_logits, original_clf_logits, batch,
                  has_edges_mask=None):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: {phase}..., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: {phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, explanation_accuracy, distillation_accuracy, fidelity, auc = self.get_eval_score(
            exp_labels, att, clf_labels, clf_logits, original_clf_logits, batch, has_edges_mask=has_edges_mask)
        desc += eval_desc

        if not batch:
            self.writer.add_scalar(f'{phase}/explanation_accuracy', explanation_accuracy, epoch)
            self.writer.add_scalar(f'{phase}/distillation_accuracy', distillation_accuracy, epoch)
            self.writer.add_scalar(f'{phase}/fidelity', fidelity, epoch)
            self.writer.add_scalar(f'{phase}/AUC', auc, epoch)

        return desc, explanation_accuracy, distillation_accuracy, fidelity, loss_dict['pred'], auc

    def get_eval_score(self, exp_labels, att, clf_labels, clf_logits, original_clf_logits, batch, has_edges_mask=None):
        distillation_accuracy = accuracy_score(clf_labels.cpu().numpy(),
                                               get_preds(clf_logits, self.multi_label).cpu().numpy())
        original_accuracy = accuracy_score(clf_labels.cpu().numpy(),
                                           get_preds(original_clf_logits, self.multi_label).cpu().numpy())
        if has_edges_mask is not None and has_edges_mask.any():
            # 只在有边的图上计算保真度
            original_accuracy_on_edged = accuracy_score(clf_labels[has_edges_mask].cpu().numpy(),
                                                        get_preds(original_clf_logits[has_edges_mask],
                                                                  self.multi_label).cpu().numpy())
            distillation_accuracy_on_edged = accuracy_score(clf_labels[has_edges_mask].cpu().numpy(),
                                                            get_preds(clf_logits[has_edges_mask],
                                                                      self.multi_label).cpu().numpy())
            fidelity = original_accuracy_on_edged - distillation_accuracy_on_edged
        else:
            # 如果没有提供mask或者所有图都没有边，则fidelity为0
            fidelity = 0.0

        if batch:
            return f'distill_acc: {distillation_accuracy:.3f}, fidelity: {fidelity:.3f}', None, None, None, None

        explanation_accuracy = 0
        if np.unique(exp_labels).shape[0] > 1:
            explanation_accuracy = roc_auc_score(exp_labels, att)

        auc = 0.0
        # 将logits转换为概率分布
        y_true = clf_labels.cpu().numpy()

        if len(np.unique(y_true)) > 1:
            if not self.multi_label:
                if clf_logits.shape[1] > 2:
                    y_scores = F.softmax(clf_logits, dim=-1).cpu().numpy()
                    try:
                        auc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
                    except ValueError:
                        pass
                elif clf_logits.shape[1] == 2:
                    y_scores = F.softmax(clf_logits, dim=-1).cpu().numpy()
                    try:
                        auc = roc_auc_score(y_true, y_scores[:, 1])
                    except ValueError:
                        pass
                elif clf_logits.shape[1] == 1:
                    y_scores = clf_logits.cpu().numpy()
                    try:
                        auc = roc_auc_score(y_true, y_scores)
                    except ValueError:
                        pass

        desc = f'distill_acc: {distillation_accuracy:.4f}, explan_acc: {explanation_accuracy:.4f}, fidelity: {fidelity:.4f}, auc: {auc:.4f}'
        return desc, explanation_accuracy, distillation_accuracy, fidelity, auc

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        if decay_interval is None or decay_interval <= 0: return final_r
        r = init_r - current_epoch // decay_interval * decay_r
        return max(r, final_r)

    def sampling(self, att_log_logits, epoch, training):
        return self.concrete_sample(att_log_logits, temp=1, training=training)

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        return src_lifted_att * dst_lifted_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']
        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_xgnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state,
                        args):
    print('=' * 80)
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size,
                                                                                    splits, random_state,
                                                                                    data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model_config['x_dim'] = x_dim
    model_config['edge_attr_dim'] = edge_attr_dim
    model_config['multi_label'] = aux_info['multi_label']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        pre_model_name = f"{data_dir}/{dataset_name}/{model_name}{random_state}.pt"
        try:
            print(f'[INFO] Loading pre-trained model from {pre_model_name}')
            model.load_state_dict(torch.load(pre_model_name)['model_state_dict'])
        except Exception as e:
            print(f'[INFO] Pre-trained model not found ({e}). Training from scratch...')
            train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                               model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
            torch.save({'model_state_dict': model.state_dict()}, pre_model_name)
    else:
        print('[INFO] Training from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if not scheduler_config else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config, **method_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    # --- !! 核心修正：在這裡添加 'mcmc_dir' 和 'pre_model_name' !! ---
    model_save_dir = data_dir / dataset_name / f'{args.log_dir}'
    pre_model_name_str = f"{dataset_name}_mt{args.multi_linear}_{model_name}_scracth{method_config['from_scratch']}_sd{random_state}"

    method_config_new = copy.deepcopy(method_config)
    method_config_new['mcmc_dir'] = model_save_dir
    method_config_new['pre_model_name'] = pre_model_name_str
    # --- 修正結束 ---

    print('[INFO] Training GSAT...')
    xgnn = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class,
                aux_info['multi_label'], random_state, method_config_new, shared_config, model_config)
    metric_dict = xgnn.train_self(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device id, -1 for cpu')
    parser.add_argument('-ld', '--log_dir', default='logs', type=str, help='')
    parser.add_argument('-mt', '--multi_linear', default=-1, type=int, help='which gmt variant to use')
    parser.add_argument('-gmt', '--gcat_multi_linear', default=-1, type=int, help='will use it to name the model')
    parser.add_argument('-st', '--sampling_trials', default=100, type=int, help='number of sampling rounds')
    parser.add_argument('-fs', '--from_scratch', default=-1, type=int, help='from scratch or not')
    parser.add_argument('-fm', '--from_mcmc', action='store_true')
    parser.add_argument('-sm', '--save_mcmc', action='store_true')
    parser.add_argument('-sd', '--seed', default=-1, type=int)
    parser.add_argument('-ie', '--info_loss_coef', default=-1, type=float)
    parser.add_argument('-r', '--ratio', default=-1, type=float)
    parser.add_argument('-ir', '--init_r', default=-1, type=float)
    parser.add_argument('-sr', '--sel_r', default=-1, type=float, help='ratio for subgraph decoding')
    parser.add_argument('-dr', '--decay_r', default=-1, type=float)
    parser.add_argument('-di', '--decay_interval', default=-1, type=int)
    parser.add_argument('-L', '--num_layers', default=-1, type=int)
    parser.add_argument('-ep', '--epochs', default=-1, type=int)
    parser.add_argument('-ft', '--force_train', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda
    method_name = 'GSAT'

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    if args.epochs >= 0:
        local_config[f'{method_name}_config']['epochs'] = args.epochs
    # ... (apply other args to local_config as in your original file)

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    seeds_to_run = range(num_seeds) if args.seed < 0 else [args.seed]

    for random_state in seeds_to_run:
        log_dir = data_dir / dataset_name / f'{args.log_dir}' / (
                time_str + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_xgnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name,
                                                       method_name, device, random_state, args)
        metric_dicts.append(metric_dict)

    print(f"\n--- Final Averaged Metrics Over {len(metric_dicts)} Seed(s) ---")

    output_filename = f"metrics_{dataset_name}_{model_name}.txt"
    with open(output_filename, "w") as f:
        f.write(
            f"--- Final Averaged Metrics for {dataset_name} with {model_name} over {len(metric_dicts)} Seed(s) ---\n")

        metric_keys = init_metric_dict.keys()

        for key in metric_keys:
            if key not in metric_dicts[0]:
                continue

            metric_values = np.array([d.get(key, 0) for d in metric_dicts])
            mean = metric_values.mean()
            std = metric_values.std()

            key_name = key.split('/')[-1]

            if 'runtime' in key_name:
                total = metric_values.sum()
                log_str = f"{key_name}: {total:.4f}s"
            elif 'time' in key_name:
                log_str = f"{key_name}: {mean:.4f}s ± {std:.4f}s"
            else:
                log_str = f"{key_name}: {mean:.4f} ± {std:.4f}"

            print(log_str)
            f.write(log_str + "\n")

    print(f"\nResults saved to {output_filename}")


if __name__ == '__main__':
    main()
