from collections import OrderedDict
from .utils.model import TRXOS
import torch
import copy
import pickle as pkl
import os


class ActionRecognizer:
    def __init__(self, input_type=None, device=None, add_hook=False, final_ckpt_path=None, seq_len=None, way=None,
                 n_joints=None, support_set_path=None):
        self.input_type = input_type
        self.device = device

        self.ar = TRXOS(None, add_hook=add_hook)
        # Fix dataparallel
        state_dict = torch.load(final_ckpt_path, map_location=torch.device(0))['model_state_dict']
        state_dict = OrderedDict({param.replace('.module', ''): data for param, data in state_dict.items()})
        self.ar.load_state_dict(state_dict)
        self.ar.cuda()
        self.ar.eval()

        self.support_set = OrderedDict()
        self.requires_focus = {}
        self.previous_frames = []
        self.seq_len = seq_len
        self.way = way
        self.n_joints = n_joints if input_type == "skeleton" else 0
        self.support_set_path = support_set_path

    def inference(self, data):
        """
        It receives an iterable of data that contains poses, images or both
        """
        if data is None or len(data) == 0:
            return {}, 0, {}

        if len(self.support_set) == 0:  # no class to predict
            return {}, 0, {}

        # Process new frame
        data = {k: torch.FloatTensor(v).cuda() for k, v in data.items()}
        self.previous_frames.append(copy.deepcopy(data))
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0, {}
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Prepare query with previous frames
        for t in list(data.keys()):
            data[t] = torch.stack([elem[t] for elem in self.previous_frames]).unsqueeze(0)
        labels = torch.IntTensor(list(range(len(self.support_set)))).unsqueeze(0).cuda()

        # Get SS
        ss = None
        ss_f = None
        if all('features' in self.support_set[c].keys() for c in self.support_set.keys()):
            ss_f = torch.stack([self.support_set[c]["features"] for c in self.support_set.keys()])  # 3 16 90
            pad = torch.zeros_like(ss_f[0]).unsqueeze(0)
            while ss_f.shape[0] < self.way:
                ss_f = torch.concat((ss_f, pad), dim=0)
            ss_f = ss_f.unsqueeze(0)  # Add batch dimension
        else:
            ss = {}
            if self.input_type in ["rgb", "hybrid"]:
                ss["rgb"] = torch.stack([self.support_set[c]["imgs"] for c in self.support_set.keys()]).unsqueeze(0)
            if self.input_type in ["skeleton", "hybrid"]:
                ss["sk"] = torch.stack([self.support_set[c]["poses"] for c in self.support_set.keys()]).unsqueeze(0)
        with torch.no_grad():
            outputs = self.ar(ss, labels, data, ss_features=ss_f)  # RGB, POSES

        # Save support features
        if ss_f is None:
            for i, s in enumerate(self.support_set.keys()):
                self.support_set[s]['features'] = outputs['support_features'][0][i]  # zero to remove batch dimension

        # Softmax
        few_shot_result = torch.softmax(outputs['logits'].squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        for k in range(len(self.support_set)):
            results[list(self.support_set.keys())[k]] = (few_shot_result[k])
        return results, open_set_result, self.requires_focus

    def remove(self, flag):
        if flag in self.support_set.keys():
            self.support_set.pop(flag)
            self.requires_focus.pop(flag)
            return True
        else:
            return False

    def train(self, inp):
        self.support_set[inp['flag']] = {c: torch.FloatTensor(inp['data'][c]).cuda() for c in inp['data'].keys()}
        self.requires_focus[inp['flag']] = inp['requires_focus']

    def save(self):
        with open(os.path.join(self.support_set_path, "support_set.pkl"), 'wb') as outfile:
            pkl.dump(self.ar.support_set, outfile)
        with open(os.path.join(self.support_set_path, "requires_focus.pkl"), 'wb') as outfile:
            pkl.dump(self.ar.requires_focus, outfile)
        return "Classes saved successfully in " + self.support_set_path

    def load(self):
        with open(os.path.join(self.support_set_path, "support_set.pkl"), 'rb') as pkl_file:
            self.support_set = pkl.load(pkl_file)
        with open(os.path.join(self.support_set_path, "requires_focus.pkl"), 'rb') as pkl_file:
            self.requires_focus = pkl.load(pkl_file)
        return f"Loaded {len(self.support_set)} classes"
