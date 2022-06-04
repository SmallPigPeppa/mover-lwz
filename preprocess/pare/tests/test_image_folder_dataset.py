import torch
import time
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

tr = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ds = ImageFolder(root='/home/mkocabas/Pictures/test', transform=tr)

for n_w in range(0,64,4):
    dataloader = DataLoader(
        dataset=ds,
        shuffle=False,
        num_workers=n_w,
        batch_size=64,
        pin_memory=True,
    )

    # start_time = time.time()
    total_start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):

        if batch_idx == 100:
            break
        # print(f'**** total {batch_idx:03d}: {time.time() - start_time:.6f}s ')

        # start_time = time.time()

    total_time = time.time() - total_start_time
    print(f'{n_w} time {total_time:.1f} s.')