from collections import defaultdict


# Used as a dict
class ImageQueue:

    def __init__(self, max_size=float('inf')):
        self.max_size = max_size
        self.client_dict = defaultdict(lambda: [])

    def put(self, client_id, image):
        self.client_dict[client_id].append(image)
        self.trim_images(client_id)

    def put_all(self, image):
        for client_id in self.client_dict:
            self.client_dict[client_id].append(image)

    def update(self, client_id, images):
        self.client_dict[client_id].extend(images)
        self.trim_images(client_id)

    def has_enough(self, client_id):
        return len(self.client_dict[client_id]) >= self.max_size

    def trim_images(self, client_id):
        if self.has_enough(client_id):
            self.client_dict[client_id] = self.client_dict[client_id][-self.max_size:]

    def get(self, client_id):
        # TODO: Fix this
        # assert len(self.client_dict[client_id]) == self.max_size
        results = self.client_dict.pop(client_id, [])
        return results

    def is_exists(self, client_id):
        return client_id in self.client_dict


class ImageList:

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.list = []

    def put(self, image):
        self.list.append(image)
        if len(self.list) > self.max_size:
            self.list = self.list[-self.max_size:]

    def get(self):
        return self.list