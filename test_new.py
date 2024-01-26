import argparse

import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report

from dataloader import MixTrafficFlowDataset4DGL
from model_new_aug import MixTemporalGNN
from utils import set_seed, get_device, mix_collate_cl_fn
from config import *


torch.autograd.set_detect_anomaly(True)


def test():
    model = MixTemporalGNN(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT, point=opt.point).to(device)
    model.load_state_dict(torch.load(config.MIX_MODEL_CHECKPOINT[:-4] + '_' + str(opt.prefix) + '.pth'))
    model.eval()
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_GRAPH_DATA,
                                        payload_path=config.TEST_GRAPH_DATA,
                                        point=opt.point)
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=False, collate_fn=mix_collate_cl_fn,
                                 num_workers=config.NUM_WORKERS, pin_memory=False)

    label_preds = []
    label_ids = []
    label_preds_packet = []
    label_ids_packet = []
    with torch.no_grad():
        for header_data, payload_data, labels, header_mask, payload_mask in dataloader:
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred, _, _, packet_out, packet_label = model(header_data, payload_data, labels, header_mask, payload_mask)
            pred_label = pred.argmax(1).detach().cpu().numpy()
            pred_label_packet = packet_out.argmax(1).detach().cpu().numpy()
            label_preds.extend(pred_label)
            label_ids.extend(labels.detach().cpu().numpy())
            label_preds_packet.extend(pred_label_packet)
            label_ids_packet.extend(packet_label.detach().cpu().numpy())

    print(classification_report(label_ids, label_preds, digits=4))
    print(classification_report(label_ids_packet, label_preds_packet, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--point", type=int, default=15)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    else:
        raise Exception('Dataset Error')

    device = get_device(index=0)
    set_seed()
    test()