import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.transform = transform
        # with open('./training/fill50k/prompt.json', 'rt') as f:
            # for line in f:
            #     self.data.append(json.loads(line))
        with open('./jeff_jhu_apl_data/full_0000_30k/ours_30000/prompt.json', 'rt') as f:
            self.data = json.load(f)['values']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # breakpoint()
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        source = cv2.imread('./' + source_filename)
        target = cv2.imread('./' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.transform:
            # print(target.shape)
            target = self.transform(target).numpy()
            target = np.transpose(target, (1, 2, 0))
            # print("After: ", target.shape)
            source = self.transform(source).numpy()
            source = np.transpose(source, (1, 2, 0))

        return dict(jpg=target, txt=prompt, hint=source)

