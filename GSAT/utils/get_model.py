import torch.nn as nn
import torch.nn.functional as F
from models import GIN, PNA, SPMotifNet, SGC
from torch_geometric.nn import InstanceNorm


def get_model(x_dim, edge_attr_dim, num_class, multi_label, model_config, device):
    if model_config['model_name'] == 'GIN':
        model = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'PNA':
        model = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'SGC':
        model = SGC(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'SPMotifNet':
        model = SPMotifNet(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    else:
        raise ValueError('[ERROR] Unknown model name!')
    return model.to(device)


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label):
        super(Criterion, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        print(f'[INFO] Using multi_label: {self.multi_label}')

    def forward(self, logits, targets):
        if self.num_class == 2 and not self.multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float().view_as(logits))
        elif self.num_class > 2 and not self.multi_label:
            loss = F.cross_entropy(logits, targets.long())
        else:
            is_labeled = targets == targets
            loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:
        preds = logits.argmax(dim=1).float()
    else:
        preds = (logits.sigmoid() > 0.5).float()
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        # --- !! 核心修改點 !! ---
        # 如果輸入的張量是空的 (例如，當一個批次中所有的圖都沒有邊時)，
        # 則直接返回，避免在 InstanceNorm 中對空張量調用 .max()
        if inputs.numel() == 0:
            return inputs
        # --- 修改結束 ---

        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)
