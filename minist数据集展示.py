import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True)

num_batches = 5
for num, (image, label) in enumerate(train_loader):
    if num>num_batches:
        break
    image_batch = torchvision.utils.make_grid(image, padding=2)
    plt.imshow(np.transpose(image_batch.numpy(), (1, 2, 0)), vmin=0, vmax=255)
    plt.show()
    print(label)
