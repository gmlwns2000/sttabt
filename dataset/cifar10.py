from torchvision import transforms, datasets
import torch

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_dataset(image_size = 32):
    transform_train = transforms.Compose([
            transforms.Resize(int(image_size * 1.2)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    transform_validation = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    transform_test = transforms.Compose([
            transforms.Resize(image_size),     
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_validation)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return {
        'train': trainset,
        'valid': validset,
        'test': testset,
    }

def load_dataloader(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, num_workers=0)
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
    }