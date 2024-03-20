from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from utils.common import create_lda_partitions
from torch.utils.data import DataLoader

dict_tranforms_train = {
    "cifar10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    ),
    "cinic10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ]
    ),
}

dict_tranforms_test = {
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    ),
    "cinic10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ]
    ),
}

# color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8,0.2)

# fedet_transform =   transforms.Compose([transforms.RandomCrop(32, padding=4),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomApply([color_jitter], p=0.8),
#                     transforms.RandomGrayscale(p=0.2),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

# dict_tranforms_kd = {"cifar10":fedet_transform}


def import_dataset(dataset, is_train=True,skip_gen_training=False,path_to_data="./data"):
    if dataset == "cifar10":
        if is_train:
            return get_cifar10_train(path_to_data=path_to_data,skip_gen_training=skip_gen_training)
        else:
            return get_cifar10_test(path_to_data)
    elif dataset == "cifar100":
        if is_train:
            return get_cifar100_train(path_to_data=path_to_data,skip_gen_training=skip_gen_training)
        else:
            return get_cifar100_test(path_to_data)
    elif dataset == "cinic10":
        if is_train:
            return get_cinic10_train(path_to_data=path_to_data,skip_gen_training=skip_gen_training)
        else:
            return get_cinic10_test(path_to_data)
    else:
        print(f"Wrong dataset name {dataset}, has it been implemented ?")
        exit(-1)


def get_dataset(path_to_data, cid, partition, transform):
    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=transform)


def get_dataloader(path_to_data, cid, is_train, batch_size, workers, transform):
    """Generates trainset/valset object and returns appropiate dataloader."""

    partition = "train" if is_train else "val"

    dataset = get_dataset(Path(path_to_data), cid, partition, transform)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "shuffle":True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_random_id_splits(total, val_ratio, shuffle=True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(
    path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0, seed=1234, is_cinic = False
):
    if is_cinic:
        images, labels = torch.load(path_to_dataset)
    else :
        images, labels = torch.load(path_to_dataset, encoding="latin1")

    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, dirichlet_dist = create_lda_partitions(
        dataset,
        num_partitions=pool_size,
        concentration=alpha,
        accept_imbalanced=True,
        seed=seed,
    )

    # Show label distribution for first partition (purely informative)

    full_histogram = []
    for part in partitions:
        hist, _ = np.histogram(part[1], bins=list(range(num_classes + 1)))
        full_histogram.append(hist)

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):
        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir, full_histogram, dirichlet_dist


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform=None,
    ):
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, torch.Tensor):
            if not isinstance(img, Image.Image):  # if not PIL image
                if not isinstance(img, np.ndarray):  # if torch tensor
                    img = img.numpy()

                img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


############################################# Datasets #############################################


def get_cifar10_train(path_to_data="./data",skip_gen_training=False):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"

    if skip_gen_training:
        print("Re-using previous generated CIFAR-10 dataset")
    else:
        # download dataset and load train set
        train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

        print("Generating unified CIFAR-10 dataset")
        # fuse all data splits into a single "training.pt"
        torch.save([train_set.data, np.array(train_set.targets)], training_data)

    # returns path where training data is and testset
    return training_data, 10, [3, 32, 32]


def get_cifar10_test(path_to_data="./data"):
    return datasets.CIFAR10(
        root=path_to_data, train=False, transform=dict_tranforms_test["cifar10"]
    )


def get_cifar100_train(path_to_data="./data",skip_gen_training=False):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    data_loc = Path(path_to_data) / "cifar-100-python"
    training_data = data_loc / "training.pt"

    if skip_gen_training:
        print("Re-using previous generated CIFAR-100 dataset")
    else:
        # download dataset and load train set
        train_set = datasets.CIFAR100(root=path_to_data, train=True, download=True)

        print("Generating unified CIFAR-10 dataset")
        # fuse all data splits into a single "training.pt"
        torch.save([train_set.data, np.array(train_set.targets)], training_data)

    # returns path where training data is and testset
    return training_data, 100, [3, 32, 32]


def get_cifar100_test(path_to_data="./data"):
    return datasets.CIFAR100(
        root=path_to_data, train=False, transform=dict_tranforms_test["cifar100"]
    )

def get_CINIC10(root):
    from torchvision.datasets.utils import download_and_extract_archive
    url_cinic = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    extract_root = Path(root) / "CINIC-10"
    download_and_extract_archive(url=url_cinic,download_root=root,extract_root=extract_root)

def get_cinic10_train(path_to_data="./data",skip_gen_training=False):
    """Downloads Cinic10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism.
    source : https://github.com/BayesWatch/cinic-10?tab=readme-ov-file and 
             https://github.com/AntonFriberg/pytorch-cinic-10"""
    
    data_loc = Path(path_to_data) / "CINIC-10"
    training_data = data_loc / "training.pt"

    if skip_gen_training:
        print("Re-using previous generated CINIC-10 dataset")
    else:
        # download dataset and load train set
        if not training_data.exists():
            get_CINIC10(root=path_to_data)
            traindir = path_to_data+"/CINIC-10/train"
            train_set = datasets.ImageFolder(root=traindir)
            print("Generating unified CINIC-10 dataset")
            trans = transforms.Compose([transforms.ToTensor()])
            train_set = datasets.ImageFolder(root=traindir,transform=trans)
            len_cinic = len(train_set)
            trainloader = DataLoader(train_set,batch_size=len_cinic)
            imgs,targets = next(iter(trainloader))
            torch.save([imgs, np.array(targets)], training_data)
            del trainloader, train_set, imgs, targets

    # returns path where training data is and testset
    return training_data, 10, [3, 32, 32]

def get_cinic10_test(path_to_data="./data"):
    """Downloads Cinic10 dataset and get test set
    source : https://github.com/BayesWatch/cinic-10?tab=readme-ov-file and 
             https://github.com/AntonFriberg/pytorch-cinic-10"""
    
    data_loc = Path(path_to_data) / "CINIC-10"
    test_dir = data_loc / "test"

    get_CINIC10(root=path_to_data)
    return datasets.ImageFolder(root=test_dir,transform=dict_tranforms_test["cinic10"])

def get_cinic10_validation(path_to_data="./data"):
    """Downloads Cinic10 dataset and get valid set
    source : https://github.com/BayesWatch/cinic-10?tab=readme-ov-file and 
             https://github.com/AntonFriberg/pytorch-cinic-10"""
    
    data_loc = Path(path_to_data) / "CINIC-10"
    test_dir = data_loc / "valid"

    get_CINIC10(root=path_to_data)
    return datasets.ImageFolder(root=test_dir,transform=dict_tranforms_test["cinic10"])