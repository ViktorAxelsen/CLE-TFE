import dgl
from dgl.data import DGLDataset

from config import Config


config = Config()


class MixTrafficFlowDataset4DGL(DGLDataset):
    def __init__(self, payload_path, header_path, point, perc=1.0):
        self.payload_path = payload_path
        self.header_path = header_path
        self.point = point
        self.perc = perc
        super(MixTrafficFlowDataset4DGL, self).__init__(name="MixTrafficFlowDataset4DGL")

    def process(self):
        self.payload_data, self.label = dgl.load_graphs(self.payload_path)
        self.header_data, self.label = dgl.load_graphs(self.header_path)
        self.label = self.label["glabel"]

        trunc_index = int(self.perc * len(self.payload_data) / config.FLOW_PAD_TRUNC_LENGTH)
        self.label = self.label[:trunc_index]
        self.payload_data = self.payload_data[:int(trunc_index * config.FLOW_PAD_TRUNC_LENGTH)]
        self.header_data = self.header_data[:int(trunc_index * config.FLOW_PAD_TRUNC_LENGTH)]

        self.payload_mask = []
        self.header_mask = []
        for sg in self.payload_data:
            if int(sg.num_nodes()) == 0 and int(sg.num_edges()) == 0:
                self.payload_mask.append(False)
            else:
                self.payload_mask.append(True)

        for sg in self.header_data:
            if int(sg.num_nodes()) == 0 and int(sg.num_edges()) == 0:
                self.header_mask.append(False)
            else:
                self.header_mask.append(True)

        assert len(self.payload_data) == len(self.header_data), "Error {} != {}".format(len(self.payload_data), len(self.header_data))

    def __getitem__(self, index):
        start_ind = config.FLOW_PAD_TRUNC_LENGTH * index
        end_ind = start_ind + config.FLOW_PAD_TRUNC_LENGTH
        return self.header_data[start_ind: start_ind + self.point], self.payload_data[start_ind: start_ind + self.point], self.label[index], \
        self.header_mask[start_ind: start_ind + self.point], self.payload_mask[start_ind: start_ind + self.point]

    def __len__(self):
        return int(len(self.payload_data) / config.FLOW_PAD_TRUNC_LENGTH)


if __name__ == '__main__':
    pass
