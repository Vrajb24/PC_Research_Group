import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import horovod.torch as hvd
import time
import os

# --- MODEL DEFINITION (ResNet-18) ---
# A standard, well-understood model suitable for CIFAR-10
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def main():
    # HVD-1: Initialize Horovod
    hvd.init()

    # HVD-2: Pin each process to a CPU core for performance.
    # This is critical on HPC systems.
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        # Pin GPU to local rank for GPU training
        torch.cuda.set_device(hvd.local_rank())

    # --- DATASET PREPARATION ---
    # Only rank 0 downloads the data
    data_dir = os.path.join(os.environ['HOME'], 'data')
    if hvd.rank() == 0:
        print("Downloading CIFAR-10 dataset...")
        os.makedirs(data_dir, exist_ok=True)
    
    # All processes wait here until rank 0 is done downloading
    hvd.broadcast(torch.tensor(0), root_rank=0, name='barrier')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                 download=(hvd.rank() == 0), transform=transform)

    # HVD-3: Use DistributedSampler to partition the dataset.
    # Each process will see a unique slice of the data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)

    # --- MODEL, OPTIMIZER, AND LOSS ---
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()

    # Scale learning rate by the number of processes.
    optimizer = optim.SGD(model.parameters(), lr=0.01 * hvd.size(), momentum=0.9)

    # HVD-4: Wrap the optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # HVD-5: Broadcast initial parameters from rank 0 to all other processes.
    # This ensures all workers start with the same model weights.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    
    # --- TRAINING LOOP ---
    num_epochs = 3 # We only need a few epochs to measure performance
    if hvd.rank() == 0:
        print(f"Starting training for {num_epochs} epochs with {hvd.size()} processes.")

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        images_processed = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            images_processed += len(data)

        # --- PERFORMANCE METRIC CALCULATION ---
        end_time = time.time()
        epoch_duration = end_time - start_time
        # Multiply by hvd.size() to get the total throughput across all workers
        images_per_sec = (images_processed / epoch_duration) * hvd.size()

        if hvd.rank() == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s | Global Img/sec: {images_per_sec:.2f}")

if __name__ == '__main__':
    main()