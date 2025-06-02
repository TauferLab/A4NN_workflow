# ---------------------------------------------------------------------------------------------------------
# Utility Functions
# Modify the return_dataloaders() function for additional datasets
# ---------------------------------------------------------------------------------------------------------


import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import timeit
import random
import torch.utils
import torch.nn as nn

# import datasets
import datasets.cifar100_dataset as my_cifar100
import datasets.cifar10_dataset as my_cifar10
import datasets.pd_dataset as my_pd

# import utilities to encode bit strings as genomes
from nsganet.models import micro_encoding
from nsganet.models import macro_encoding

# import evolutionary models
from nsganet.models.macro_models import EvoNetwork
from nsganet.models.micro_models import NetworkCIFAR as Network

g = torch.Generator()
g.manual_seed(0)

# for dataloader reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):

    n_params_from_auxiliary_head = np.sum(np.prod(v.size()) for name, v in model.named_parameters()) - \
                                   np.sum(np.prod(v.size()) for name, v in model.named_parameters()
                                          if "auxiliary" not in name)
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (n_params_trainable - n_params_from_auxiliary_head) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

# this save is just for simple weights and is not portable
def save(model, model_path):
    torch.save(model.state_dict(), model_path)

# this load is just for simple weights and is not portable 
def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    
    print('Experiment dir : {}'.format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# Gets image shape and color channels from first image in dataset
def get_image_dim(DIRECTORY, EXT):
    images = os.scandir(DIRECTORY)
    files = []    
    for i in images:
        if i.name.endswith("."+EXT) and not i.name.startswith("."):
            files.append(i.path)
            break
    
    # Open one image to get dimensions from
    im = Image.open(files[0]) 
    c = len(im.getbands())
    
    return im.size, c

def assign_gpu(how='random'):
    # this is for if you have multiple gpus per node
    if how == 'random':
        num_gpus = torch.cuda.device_count() - 1
        if num_gpus == 0:
            return 0
        else:
            gpu = random.randint(0, num_gpus)
    else:
        raise NotImplementedError("there are no options besides random assignment at the moment.")
    return gpu 

def num_gpus():
    return torch.cuda.device_count()

def gpu_usage(device):
    start = timeit.timeit()
    print(torch.cuda.utilization(device))
    print(torch.cuda.mem_get_info(device))
    print(torch.cuda.memory_summary(device))
    print(torch.cuda.memory_usage(device))
    end = timeit.timeit()
    print(end - start)
    return


#-----------------------------------------------------------------------#
# Utility functions to define/create dataset objects and model objects
#-----------------------------------------------------------------------#
def return_dataloaders(dataset_name, 
                       data_root=None, 
                       batch_size=128,
                       cutout=False, 
                       cutout_length=16):

    """
    Function returns the dataset objects for train and validation data
    """
    # initialize args
    data_args = {}


     # ---- Dataset options ------------- #
    if dataset_name == 'CIFAR-100':
        NUM_CLASSES = 100
        im_channels = 3 # the image dimension
        data_shape = (32, 32) # image input shape

        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2675, 0.2565, 0.2761]

        # Performs transformations to images (if working with other datasets then values must be changed accordingly)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))

        train_transform.transforms.append(transforms.Normalize(MEAN, STD))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        train_data = my_cifar100.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        valid_data = my_cifar100.CIFAR100(root=data_root, train=False, download=True, transform=valid_transform)
    elif dataset_name == 'CIFAR-10':
        NUM_CLASSES = 10
        im_channels=3 # the image dimension
        data_shape = (32, 32) # image input shape

        # data_root = data_root + dataset.lower()
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))

        train_transform.transforms.append(transforms.Normalize(MEAN, STD))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        train_data = my_cifar10.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        valid_data = my_cifar10.CIFAR10(root=data_root, train=False, download=True, transform=valid_transform)
    elif dataset_name == 'PD':
        NUM_CLASSES = 2
        image = f'{data_root}/images/trainset/' # path to pd images
        data_shape, im_channels = get_image_dim(image, EXT='tiff') # Change EXT according to your data

        train_transform = transforms.ToTensor()
        valid_transform = transforms.ToTensor()

        train_data = my_pd.ProteinDataset(data_root, train=True)
        valid_data = my_pd.ProteinDataset(data_root, train=False)
    else:
        raise NotImplementedError("Only CIFAR-10 and CIFAR-100 supported. Add code here for your custom datasets...")

    data_args['im_channels'] = im_channels
    data_args['num_classes'] = NUM_CLASSES
    data_args['data_shape'] = data_shape

    # ---- Create data loaders --- # 
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=g)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=g)

    return train_queue, valid_queue, data_args


def return_architecture(search_space, genome, data_args=None, init_channels=24, micro_layers=11, micro_auxiliary=False):
    """
    Function to return evolutionary architecture
    """
    # retrieve data related arguments
    im_channels = data_args['im_channels']
    NUM_CLASSES = data_args['num_classes']
    data_shape = data_args['data_shape'] 

    # create the genome encoding and call the appropriate class for creating the network
    if search_space == 'micro': 
        # micro encoding performs a search inside the CNN layer 
        genotype = micro_encoding.decode(genome)
        model = Network(init_channels, NUM_CLASSES, micro_layers, micro_auxiliary, genotype)
    elif search_space == 'macro':
        # macro encoding performs a search at the level of CNN layer connections
        genotype = macro_encoding.decode(genome)

        # Create a list of in/out channels based on the number of phases
        # double the channels as we add more phases
        num_phases = len(genotype)
        channels = []
        for i in range(num_phases):
            if i == 0:
                channels.append((im_channels, init_channels))
            else:
                channels.append((init_channels, init_channels*2))
                init_channels*=2 # double channels

        # create the model
        model = EvoNetwork(genotype, channels, NUM_CLASSES, data_shape, decoder='residual')
    else:
        raise NameError('Unknown search space type')
    
    return model, genotype


#-----------------------------------------------------------------------#
# Utility functions for training/evaluating NN models
#-----------------------------------------------------------------------#
def train_and_val(train_queue, model, criterion, optimizer, train_params, valid_queue, device):
    train_acc, train_loss = train(train_queue, model, criterion, optimizer, train_params, device)
    val_acc, val_loss = infer(valid_queue, model, criterion, device)
    return train_acc, train_loss, val_acc, val_loss

def train(train_queue, net, criterion, optimizer, params, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    queue_length = len(train_queue)

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if params['auxiliary']:
            loss_aux = criterion(outputs_aux, targets)
            loss += params['auxiliary_weight'] * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.*correct/total, train_loss/total


def infer(valid_queue, net, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total

    return acc, test_loss/total