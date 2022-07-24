```python
import torch
import time
from tqdm import tqdm

class KMEANS:
    def __init__(self, n_clusters=5, max_iter=10):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.count = 0

    def fit(self, x):
        # 随机选择初始中心点
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)) ###[20]
        init_points = x[init_row]
        self.centers = init_points ##[20,128],
        while self.count < self.max_iter:
            # 聚类标记
            self.nearest_center(x)
            print(self.labels)
            # 更新中心点
            self.update_center(x)
            self.count += 1

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long() ##[20]
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
        self.labels = labels ##聚类标记

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1]))
        for i in range(self.n_clusters):
            cluster_samples = x[self.labels == i]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers





if __name__ == "__main__":
  kmeans = KMEANS()
  x = torch.randn((20,128))
  kmeans.fit(x)

```

