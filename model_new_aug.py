import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import SAGEConv


class GCN(nn.Module):
    def __init__(self, embedding_size, h_feats, dropout):
        super(GCN, self).__init__()
        self.gcn_out_dim = 4 * h_feats
        self.embedding = nn.Embedding(256 + 1, embedding_size)
        self.gcn1 = SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn2 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn3 = SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
        self.gcn4 = SAGEConv(h_feats, h_feats, 'mean', activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))

    def forward(self, g, in_feat):
        in_feat = in_feat.long()
        h = self.embedding(in_feat.view(-1))
        h1 = self.gcn1(g, h)
        h2 = self.gcn2(g, h1)
        h3 = self.gcn3(g, h2)
        h4 = self.gcn4(g, h3)
        g.ndata['h'] = torch.cat((h1, h2, h3, h4), dim=1)
        g_vec = dgl.mean_nodes(g, 'h')

        return g_vec


class Cross_Gated_Info_Filter(nn.Module):
    def __init__(self, in_size, point):
        super(Cross_Gated_Info_Filter, self).__init__()
        self.filter1 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(point),
            nn.Linear(in_size, in_size)
        )
        self.filter2 = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.PReLU(point),
            nn.Linear(in_size, in_size)
        )

    def forward(self, x, y):
        ori_x = x
        ori_y = y
        z1 = self.filter1(x).sigmoid() * ori_y
        z2 = self.filter2(y).sigmoid() * ori_x

        return torch.cat([z1, z2], dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MixTemporalGNN(nn.Module):
    def __init__(self, num_classes, embedding_size=64, h_feats=128, dropout=0.2, downstream_dropout=0.0,
                 point=15, seq_aug_ratio=0.8, drop_edge_ratio=0.1, drop_node_ratio=0.1, K=15,
                 hp_ratio=0.5, tau=0.07, gtau=0.07):
        super(MixTemporalGNN, self).__init__()
        self.header_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.payload_graphConv = GCN(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.gcn_out_dim = 4 * h_feats
        self.point = point
        self.seq_aug_ratio = seq_aug_ratio
        self.drop_edge_ratio = drop_edge_ratio
        self.drop_node_ratio = drop_node_ratio
        self.K = K
        self.hp_ratio = hp_ratio
        self.gated_filter = Cross_Gated_Info_Filter(in_size=self.gcn_out_dim, point=self.point)
        self.rnn = nn.LSTM(input_size=self.gcn_out_dim * 2, hidden_size=self.gcn_out_dim * 2, num_layers=2, bidirectional=True, dropout=downstream_dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 4, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim)
        )
        self.cls = nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        self.packet_head = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 2, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim),
            nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        )

        self.supcl = SupConLoss(temperature=tau, base_temperature=tau)
        self.supcl_g = SupConLoss(temperature=gtau, base_temperature=gtau)
        self.drop_edge_trans = dgl.DropEdge(p=self.drop_edge_ratio)
        self.drop_node_trans = dgl.DropNode(p=self.drop_node_ratio)

    def forward(self, header_graph_data, payload_graph_data, labels, header_mask, payload_mask):
        header_mask = header_mask.reshape(labels.shape[0], self.point, -1)[:, :self.K, :].reshape(-1)
        payload_mask = payload_mask.reshape(labels.shape[0], self.point, -1)[:, :self.K, :].reshape(-1)

        aug_header_graph_data = self.drop_node_trans(self.drop_edge_trans(header_graph_data))
        aug_payload_graph_data = self.drop_node_trans(self.drop_edge_trans(payload_graph_data))
        header_gcn_out = self.header_graphConv(header_graph_data, header_graph_data.ndata['feat']).reshape(
            labels.shape[0], self.point, -1)
        payload_gcn_out = self.payload_graphConv(payload_graph_data, payload_graph_data.ndata['feat']).reshape(
            labels.shape[0], self.point, -1)
        aug_header_gcn_out = self.header_graphConv(aug_header_graph_data, aug_header_graph_data.ndata['feat']).reshape(
            labels.shape[0], self.point, -1)
        aug_payload_gcn_out = self.payload_graphConv(aug_payload_graph_data, aug_payload_graph_data.ndata['feat']).reshape(
            labels.shape[0], self.point, -1)

        temp1 = header_gcn_out[:, :self.K, :].reshape(-1, header_gcn_out.shape[2])[header_mask]
        temp2 = aug_header_gcn_out[:, :self.K, :].reshape(-1, aug_header_gcn_out.shape[2])[header_mask]
        mask12 = torch.any(temp1 != 0, dim=1) & torch.any(temp2 != 0, dim=1)
        temp3 = payload_gcn_out[:, :self.K, :].reshape(-1, payload_gcn_out.shape[2])[payload_mask]
        temp4 = aug_payload_gcn_out[:, :self.K, :].reshape(-1, aug_payload_gcn_out.shape[2])[payload_mask]
        mask34 = torch.any(temp3 != 0, dim=1) & torch.any(temp4 != 0, dim=1)

        header_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[header_mask]
        payload_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[payload_mask]

        header_cl_loss = self.supcl_g(torch.cat((F.normalize(temp1[mask12], p=2).unsqueeze(1), F.normalize(temp2[mask12], p=2).unsqueeze(1)), dim=1), header_label[mask12])
        payload_cl_loss = self.supcl_g(torch.cat((F.normalize(temp3[mask34], p=2).unsqueeze(1), F.normalize(temp4[mask34], p=2).unsqueeze(1)), dim=1), payload_label[mask34])
        graph_cl_loss = self.hp_ratio * header_cl_loss + (1 - self.hp_ratio) * payload_cl_loss

        gcn_out = self.gated_filter(header_gcn_out, payload_gcn_out)

        # packet-level head
        packet_mask = header_mask & payload_mask
        packet_rep = gcn_out[:, :self.K, :].reshape(-1, gcn_out.shape[2])[packet_mask]
        packet_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[packet_mask]
        packet_out = self.packet_head(packet_rep)

        gcn_out_aug = self.gated_filter(aug_header_gcn_out, aug_payload_gcn_out)
        aug_index = []
        for _ in range(len(gcn_out_aug)):
            index = np.random.choice(range(self.point), size=int(self.point * self.seq_aug_ratio), replace=False)
            index.sort()
            aug_index.append(index)

        aug_index = torch.tensor(np.array(aug_index), dtype=int, device=gcn_out.device)
        aug_index = aug_index.unsqueeze(2)
        aug_index = aug_index.repeat(1, 1, gcn_out_aug.shape[2])
        gcn_out_aug = torch.gather(gcn_out_aug, dim=1, index=aug_index)

        gcn_out = gcn_out.transpose(0, 1)
        _, (h_n, _) = self.rnn(gcn_out)
        rnn_out = torch.cat((h_n[-1], h_n[-2]), dim=1)
        gcn_out_aug = gcn_out_aug.transpose(0, 1)
        _, (h_n_aug, _) = self.rnn(gcn_out_aug)
        rnn_out_aug = torch.cat((h_n_aug[-1], h_n_aug[-2]), dim=1)
        rnn_out = F.normalize(rnn_out, p=2)
        rnn_out_aug = F.normalize(rnn_out_aug, p=2)
        cl_loss = self.supcl(torch.cat((rnn_out.unsqueeze(1), rnn_out_aug.unsqueeze(1)), dim=1), labels)
        out = self.fc(rnn_out)
        out = self.cls(out)

        return out, cl_loss, graph_cl_loss, packet_out, packet_label


if __name__ == '__main__':
    pass