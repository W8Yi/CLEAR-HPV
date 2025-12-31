import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        # print('device', device)
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # print device of all_preds
        # print('all_preds device', all_preds.device)
        # print('all_targets device', all_targets.device)
        # print('logits device', logits.device)
        # print('all_instances device', all_instances.device)
   
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        # print('label:', label)
        # print('label shape:', label.shape)
        # print('label type:', type(label))
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            # print('inst_labels:', inst_labels)
            # print('inst_labels shape:', inst_labels.shape)
            # print('inst_labels type:', type(inst_labels))
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
class CLAM_MH(nn.Module):
    """
    True multi-head attention MIL.
    - attention_net outputs H heads over instances (A: H x N after transpose)
    - Per-head pooled features M: H x D
    - Aggregate heads by 'concat' (default), 'mean', or 'max' to get bag feature
    - Final classifier predicts n_classes (binary for HPV)
    - Instance clustering uses the MEAN attention across heads to pick top/bottom-K
      so your existing instance loss flow remains compatible.
    """
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,               # binary HPV: 2
        n_heads: int = 4,                 # number of attention heads
        head_pool: str = "concat",        # 'concat' | 'mean' | 'max'
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
        embed_dim: int = 1024,
    ):
        super().__init__()
        assert head_pool in ("concat", "mean", "max")
        self.head_pool = head_pool
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.n_heads = n_heads

        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]  # [in_dim, mid_dim, attn_hidden]

        # backbone + attention heads
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_heads)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_heads)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # instance classifiers for clustering (same shape as CLAM_SB/MB: per class)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # bag classifier after head aggregation
        if self.head_pool == "concat":
            feat_dim = size[1] * n_heads
        else:
            feat_dim = size[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def _inst_eval_mean_heads(self, A_heads, h, classifier):
        """
        Instance selection using mean attention across heads.
        A_heads: H x N (after transpose & softmax across N for each head)
        h: N x D
        """
        device = h.device
        if len(A_heads.shape) == 1:
            A_heads = A_heads.view(1, -1)  # not expected here
        A_mean = A_heads.mean(dim=0, keepdim=True)  # 1 x N
        # top-K positive and negative by mean attention
        top_p_ids = torch.topk(A_mean, self.k_sample)[1][-1]          # (k,)
        top_n_ids = torch.topk(-A_mean, self.k_sample, dim=1)[1][-1]  # (k,)

        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)  # (2k, D)

        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def _inst_eval_out_mean_heads(self, A_heads, h, classifier):
        """
        Out-of-class branch: pick top-K by mean attention and label them negative.
        """
        device = h.device
        A_mean = A_heads.mean(dim=0, keepdim=True)  # 1 x N
        top_p_ids = torch.topk(A_mean, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        h: N x in_dim
        returns: logits, Y_prob, Y_hat, A_raw, results_dict
        """
        # attention over mid features
        A, h_mid = self.attention_net(h)          # (N x H), (N x mid_dim)
        A = torch.transpose(A, 1, 0)              # (H x N)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)                   # softmax over N for each head

        # instance-level clustering (mean across heads for selection)
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self._inst_eval_mean_heads(A, h_mid, classifier)
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self._inst_eval_out_mean_heads(A, h_mid, classifier)
                    else:
                        continue
                total_inst_loss += instance_loss
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        # per-head pooled features: M = A @ h_mid  => (H x D)
        M_heads = torch.mm(A, h_mid)              # (H x mid_dim)

        # aggregate heads -> bag feature
        if self.head_pool == "concat":
            M_agg = M_heads.reshape(1, -1)        # (1 x H*D)
        elif self.head_pool == "mean":
            M_agg = M_heads.mean(dim=0, keepdim=True)   # (1 x D)
        else:  # 'max'
            M_agg, _ = M_heads.max(dim=0, keepdim=True) # (1 x D)

        # final logits
        logits = self.classifier(M_agg)           # (1 x n_classes)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({'features': M_agg})

        return logits, Y_prob, Y_hat, A_raw, results_dict
    
class ABMIL_SB(nn.Module):
    """
    Projection + ABMIL attention, one attention map (K=1).
    CLAM-compatible forward: returns (logits, Y_prob, Y_hat, A_raw, results_dict)
    """
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        n_classes: int = 2,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        in_dim, mid_dim, attn_dim = self.size_dict[size_arg]

        # projection to mid_dim
        fc = [nn.Linear(in_dim, mid_dim), nn.ReLU(), nn.Dropout(dropout)]
        # single attention head K=1
        attn = Attn_Net_Gated(L=mid_dim, D=attn_dim, dropout=dropout, n_classes=1) if gate \
               else Attn_Net(L=mid_dim, D=attn_dim, dropout=dropout, n_classes=1)
        fc.append(attn)
        self.attention_net = nn.Sequential(*fc)

        # slide classifier on pooled mid features
        self.classifier = nn.Linear(mid_dim, n_classes)

    def forward(self, h, label=None, instance_eval: bool = False,
                return_features: bool = False, attention_only: bool = False):
        """
        h: N x embed_dim
        """
        A, h_mid = self.attention_net(h)        # A: N x 1, h_mid: N x mid_dim
        A = A.transpose(1, 0)                   # 1 x N
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)                 # softmax over instances

        # pooled bag feature: 1 x mid_dim
        M = torch.mm(A, h_mid)

        logits = self.classifier(M)             # 1 x n_classes
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results = {}
        if return_features:
            results["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results

class TransMIL_SB(nn.Module):
    """
    Single-branch TransMIL-style MIL with projection.
    CLAM-compatible forward signature and return tuple:
      (logits, Y_prob, Y_hat, A_raw, results_dict)
    - Uses a Transformer encoder over instance tokens.
    - Pools with a CLS→tokens multihead attention to expose a 1×N attention map.
    """
    def __init__(
        self,
        size_arg: str = "small",          # matches CLAM size choices
        dropout: float = 0.0,
        n_classes: int = 2,
        embed_dim: int = 1024,            # input feature dim
        d_model: int = 256,               # transformer width (kept = CLAM mid-dim)
        n_heads: int = 1,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        attn_pool_heads: int = 4,         # heads used only for CLS→tokens pooling
    ):
        super().__init__()
        # projection to d_model (same role as CLAM mid feature)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # transformer encoder over tokens (batch_first=True for [B, S, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # learned CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # pooling via CLS→tokens multi-head attention to expose a 1×N map
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attn_pool_heads,
            dropout=dropout,
            batch_first=True,
        )

        # classifier on pooled bag feature
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        h,                     # N × embed_dim
        label=None,
        instance_eval: bool=False,
        return_features: bool=False,
        attention_only: bool=False,
    ):
        # N x Din -> N x d_model
        H = self.proj(h)

        # prepend CLS: (1 x (N+1) x d_model)
        cls = self.cls_token.expand(1, -1, -1)           # 1 x 1 x d_model
        tokens = H.unsqueeze(0)                           # 1 x N x d_model
        x = torch.cat([cls, tokens], dim=1)               # 1 x (N+1) x d_model

        # transformer encoding
        z = self.encoder(x)                               # 1 x (N+1) x d_model
        z_cls, z_tokens = z[:, :1, :], z[:, 1:, :]        # 1 x 1 x d_model, 1 x N x d_model

        # CLS→tokens attention pooling (query=CLS, key/value=tokens)
        pooled, attn = self.pool_mha(
            query=z_cls, key=z_tokens, value=z_tokens, need_weights=True, average_attn_weights=False
        )
        # attn: shape (1, heads, 1, N) if average_attn_weights=False in newer PyTorch,
        # or (1, 1, N) if averaged. Make it 1 x N either way.
        if attn.dim() == 4:
            A = attn.mean(dim=1).squeeze(1)              # 1 x N (mean over heads)
        else:
            A = attn                                      # 1 x N

        if attention_only:
            return A                                      # 1 x N

        A_raw = A.clone()
        A = F.softmax(A, dim=1)                           # normalize over instances

        # pooled bag feature via attention weights (equivalent to pooled)
        M = torch.bmm(A.unsqueeze(1), z_tokens).squeeze(1)  # 1 x d_model

        # classification
        logits = self.classifier(M)                       # 1 x n_classes
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results = {}
        if return_features:
            results["features"] = M
        # instance_eval not used in SB; kept for API parity

        return logits, Y_prob, Y_hat, A_raw, results
    


class ABMIL_CLAM(nn.Module):
    """
    ABMIL + CLAM-style instance-level clustering (Option A).
    Single-head attention; instance selection uses that head directly.
    API identical to CLAM_SB.
    """
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        n_classes: int = 2,
        k_sample: int = 8,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        # ABMIL dimensions identical to CLAM_SB
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        in_dim, mid_dim, attn_dim = self.size_dict[size_arg]

        # mid projection
        fc = [nn.Linear(in_dim, mid_dim), nn.ReLU(), nn.Dropout(dropout)]

        # 1 attention head
        if gate:
            attn = Attn_Net_Gated(L=mid_dim, D=attn_dim, dropout=dropout, n_classes=1)
        else:
            attn = Attn_Net(L=mid_dim, D=attn_dim, dropout=dropout, n_classes=1)
        fc.append(attn)

        self.attention_net = nn.Sequential(*fc)

        # slide classifier
        self.classifier = nn.Linear(mid_dim, n_classes)

        # instance classifiers (one per class)
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(mid_dim, 2) for _ in range(n_classes)
        ])

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def _inst_eval(self, A, h_mid, classifier):
        """
        CLAM-SB instance selection: use A (1 x N) directly.
        """
        device = h_mid.device
        A = A.view(1, -1)

        # Top-K and bottom-K
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]       # (k,)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]

        top_p = h_mid[top_p_ids]
        top_n = h_mid[top_n_ids]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        logits = classifier(all_instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        loss = self.instance_loss_fn(logits, all_targets)
        return loss, preds, all_targets

    def _inst_eval_out(self, A, h_mid, classifier):
        """
        Out-of-class branch: top-K by A but all labeled negative.
        """
        device = h_mid.device
        A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = h_mid[top_p_ids]

        p_targets = self.create_negative_targets(self.k_sample, device)

        logits = classifier(top_p)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        loss = self.instance_loss_fn(logits, p_targets)
        return loss, preds, p_targets

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        
        A, h_mid = self.attention_net(h)    # A: N x 1
        A = A.transpose(1, 0)               # 1 x N

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)             # normalize over N

        # --------------------------
        # Instance-level clustering
        # --------------------------
        if instance_eval:
            total_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i in range(self.n_classes):
                classifier = self.instance_classifiers[i]
                if inst_labels[i].item() == 1:
                    loss, preds, targets = self._inst_eval(A, h_mid, classifier)
                else:
                    if self.subtyping:
                        loss, preds, targets = self._inst_eval_out(A, h_mid, classifier)
                    else:
                        continue

                total_loss += loss
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            if self.subtyping:
                total_loss /= self.n_classes

        # pooled slide rep: 1 x mid_dim
        M = torch.mm(A, h_mid)

        logits = self.classifier(M)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        results = {}
        if instance_eval:
            results = {
                "instance_loss": total_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        if return_features:
            results["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results
    

class TransMIL_CLAM(nn.Module):
    """
    TransMIL + CLAM-style instance clustering.
    Uses the single CLS→tokens attention vector A (1 x N) directly (Option A).
    """
    def __init__(
        self,
        size_arg: str = "small",
        dropout: float = 0.0,
        n_classes: int = 2,
        k_sample: int = 8,
        embed_dim: int = 1024,
        d_model: int = 256,
        n_heads: int = 1,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        attn_pool_heads: int = 4,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping: bool = False,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        # projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # learned CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # CLS → tokens attention pooling
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attn_pool_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Linear(d_model, n_classes)

        # per-class instance classifiers
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(d_model, 2) for _ in range(n_classes)
        ])

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def _inst_eval(self, A, h_mid, classifier):
        device = h_mid.device
        A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]

        top_p = h_mid[top_p_ids]
        top_n = h_mid[top_n_ids]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        logits = classifier(all_instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        loss = self.instance_loss_fn(logits, all_targets)
        return loss, preds, all_targets

    def _inst_eval_out(self, A, h_mid, classifier):
        device = h_mid.device
        A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = h_mid[top_p_ids]

        p_targets = self.create_negative_targets(self.k_sample, device)

        logits = classifier(top_p)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        loss = self.instance_loss_fn(logits, p_targets)
        return loss, preds, p_targets

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):

        # mid projection
        H = self.proj(h)                       # N x d_model
        tokens = H.unsqueeze(0)                # 1 x N x d_model

        # add CLS
        cls = self.cls_token.expand(1, -1, -1) # 1 x 1 x d_model
        x = torch.cat([cls, tokens], dim=1)    # 1 x (N+1) x d_model

        # transformer encoding
        z = self.encoder(x)
        z_cls, z_tokens = z[:, :1, :], z[:, 1:, :]   # 1 x 1 x d_model, 1 x N x d_model

        # CLS → tokens attention pooling
        pooled, attn = self.pool_mha(
            query=z_cls, key=z_tokens, value=z_tokens,
            need_weights=True, average_attn_weights=False
        )

        # convert attn to shape 1 x N
        if attn.dim() == 4:
            A = attn.mean(dim=1).squeeze(1)     # mean across heads
        else:
            A = attn                             # already 1 x N

        if attention_only:
            return A

        A_raw = A.clone()
        A = F.softmax(A, dim=1)                 # normalize over N

        # --------------------------
        # Instance-level clustering
        # --------------------------
        if instance_eval:
            total_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i in range(self.n_classes):
                classifier = self.instance_classifiers[i]
                if inst_labels[i].item() == 1:
                    loss, preds, targets = self._inst_eval(A, z_tokens.squeeze(0), classifier)
                else:
                    if self.subtyping:
                        loss, preds, targets = self._inst_eval_out(A, z_tokens.squeeze(0), classifier)
                    else:
                        continue

                total_loss += loss
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            if self.subtyping:
                total_loss /= self.n_classes

        # pooled bag feature = pooled (1 x d_model)
        M = pooled                               # 1 x d_model

        logits = self.classifier(M)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        results = {}
        if instance_eval:
            results = {
                "instance_loss": total_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        if return_features:
            results["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results