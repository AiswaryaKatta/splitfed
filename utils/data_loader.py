import torchvision
import torchvision.transforms as transforms

def load_mnist(batch_size=32, quick_mode=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if quick_mode:
        train_set.data = train_set.data[:2000]
        train_set.targets = train_set.targets[:2000]
        test_set.data = test_set.data[:500]
        test_set.targets = test_set.targets[:500]

    return train_set, test_set
