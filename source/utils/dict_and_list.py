from collections import defaultdict
import numpy as np


class WaitingImageQueue:

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.client_dict = defaultdict(lambda: [])

    def put(self, client_id, image):
        self.client_dict[client_id].append(image)

    def has_enough(self, client_id):
        return len(self.client_dict[client_id]) >= self.max_size

    def get(self, client_id):
        # TODO: Fix this
        # assert len(self.client_dict[client_id]) == self.max_size
        results = self.client_dict[client_id]
        self.client_dict[client_id] = []
        return results

    def get_all(self):
        results = [(k,v) for k,v in self.client_dict.items()]
        self.client_dict = defaultdict(lambda: [])
        return results


class EmbsDict():

    def __init__(self, max_size=20):
        self.embs_dict = defaultdict(lambda: None)
        self.max_size = max_size
        self.step_size = max_size // 2

    def put(self, client_id, new_embs):
        # embs: (5, 128)
        if self.embs_dict[client_id] is None:
            self.embs_dict[client_id] = new_embs
        else:
            old_embs = self.embs_dict[client_id]
            if len(old_embs) > self.max_size:
                old_embs = old_embs[self.step_size:]
            self.embs_dict[client_id] = np.vstack((old_embs, new_embs))

    def get(self, client_id):
        return self.embs_dict[client_id]

    def pop(self, client_id):
        return self.embs_dict.pop(client_id, np.array([]))
