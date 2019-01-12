from torchvision import transforms, datasets
import torch

BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
WORKERS = 8
NO_CUDA = False

# image normalization
IMAGE_CROP = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_folder_loaders(folder, dataset, name, batch=BATCH_SIZE, rgb=True, depth=False):
    # Check for CUDA usage
    cuda = not NO_CUDA and torch.cuda.is_available()
    
    # data pre-processing
    workers = WORKERS if cuda else 0

    if rgb and depth:
        data_transform = get_data_transform(name, True)
        data_transform_2 = get_data_transform(name, False)
        data_transform = (data_transform, data_transform_2)
    elif rgb:
        data_transform = get_data_transform(name, True)
    elif depth:
        data_transform = get_data_transform(name, False)
    else:
        raise(Exception("rgb and depth are both False"))
    
    # Build the training loader
    try:
        dataset = dataset(root=folder, transform=data_transform, rgb=rgb, depth=depth)
    except:
        dataset = dataset(root=folder, transform=data_transform)
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    return loader


# DATA_TRANSFORMS = ["NO", "SC", "SM", "MI"]
def get_data_transform(name, rgb):
    # Create Data loader w.r.t. chosen transformations
    if rgb:
        MEAN = IMAGENET_MEAN
        STD = IMAGENET_STD
    else:
        MEAN = IMAGENET_MEAN
        STD = IMAGENET_STD
    
    if name == "NO":
        data_transform = transforms.Compose([
            transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    elif name == "SC":  # scaling
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    elif name == "SM":  # scaling and mirror
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    elif name == "MI":  # mirror
        data_transform = transforms.Compose([
            transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        raise (Exception("Transform code not known"))
        
    return data_transform
