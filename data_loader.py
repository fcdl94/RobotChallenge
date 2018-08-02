from torchvision import transforms, datasets
import torch

BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
WORKERS = 8
NO_CUDA = False

# image normalization
IMAGE_CROP = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
#IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_STD = [1, 1, 1]


def get_image_folder_loaders(folder, dataset, mirror=True, scaling=True, batch=BATCH_SIZE):
    # Check for CUDA usage
    cuda = not NO_CUDA and torch.cuda.is_available()
    
    # data pre-processing
    workers = WORKERS if cuda else 0
    
    data_transform = get_data_transform(mirror, scaling)
    # Build the training loader
    dataset = dataset(root=folder, transform=data_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    return loader


def get_data_transform(mirror, scaling):
    # Create Data loader w.r.t. chosen transformations
    if mirror:
        if scaling:
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    else:
        if scaling:
            data_transform = transforms.Compose([
                # transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    return data_transform
