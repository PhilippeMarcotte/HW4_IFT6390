from __future__ import print_function

import argparse
import sys

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kdataset import *
from utils import *
from pycrayon import *
# Random seed

model_names = sorted(name for name in nnmodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(nnmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Ensembler')

print("Available models:" + str(model_names))

parser.add_argument('--validationRatio', type=float, default=0.10, help='test Validation Split.')
parser.add_argument('--optim', type=str, default='adam', help='Adam or SGD')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--batch_size', default=30, type=int, metavar='N', help='train batchsize')

parser.add_argument('--num_classes', type=int, default=12, help='Number of Classes in data set.')
parser.add_argument('--data_path', default='../data/', type=str, help='Path to train dataset')
parser.add_argument('--data_path_test', default='../data/', type=str, help='Path to test dataset')
parser.add_argument('--dataset', type=str, default='quickdraw', choices=['seeds', 'quickdraw'], help='Choose between data sets')

# parser.add_argument('--arch', metavar='ARCH', default='simple', choices=model_names)
parser.add_argument('--imgDim', default=1, type=int, help='number of Image input dimensions')
parser.add_argument('--img_scale', default=80, type=int, help='Image scaling dimensions')
parser.add_argument('--base_factor', default=20, type=int, help='SENet base factor')

parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--current_time', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    help='Current time.')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.005, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.95, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
# parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# Checkpoints
parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--save_path_model', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
# random seed
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')

# Random Erasing
from ktransforms import *
parser.add_argument('--p', default=0.32, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# Use CUDA
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
use_cuda = args.use_cuda

if args.manualSeed is None:
    args.manualSeed = 999
fixSeed(args)


def train(train_loader, model, criterion, optimizer, args):
    if args.use_cuda:
        model.cuda()
        criterion.cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            images, target = images.cuda(), target.cuda()
            images, target = Variable(images), Variable(target)
        # compute y_pred
        y_pred = model(images)
        loss = criterion(y_pred, target)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target.data, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        acc.update(prec1.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
        #if i % 400 == 0:
            print('TRAIN: LOSS-->{loss.val:.4f} ({loss.avg:.4f})\t' 'ACC-->{acc.val:.3f}% ({acc.avg:.3f}%)'.format(loss=losses, acc=acc))
            if use_tensorboard:
                exp.add_scalar_value('tr_epoch_loss', losses.avg, step=epoch)
                exp.add_scalar_value('tr_epoch_acc', acc.avg, step=epoch)

    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))

def validate(val_loader, model, criterion, args):
    if args.use_cuda:
        model.cuda()
        criterion.cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

            # compute y_pred
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            # measure accuracy and record loss
            prec1, temp_var = accuracy(y_pred.data, labels.data, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 400 == 0:
                print('VAL:   LOSS--> {loss.val:.4f} ({loss.avg:.4f})\t''ACC-->{acc.val:.3f} ({acc.avg:.3f})'.format(
                    loss=losses, acc=acc))

            if i % 50 == 0:
                if use_tensorboard:
                    exp.add_scalar_value('val_epoch_loss', losses.avg, step=epoch)
                    exp.add_scalar_value('val_epoch_acc', acc.avg, step=epoch)

    print('FINAL ACC: {acc.avg:.3f}'.format(acc=acc))
    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))


def loadDB(args):
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    dataset = QuickDrawDataset(args.data_path, split='train', transform=train_transforms)

    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - int(args.validationRatio * len(dataset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_indices = indices[len(indices) - int(args.validationRatio * len(dataset)):]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    t_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    v_loader = ValidationQuickDrawLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)

    dataset_sizes = {
        'train': len(t_loader.dataset),
        'valid': len(v_loader.dataset)
    }
    print(dataset_sizes)
    print('#Classes: {}'.format(len(dataset.classes)))
    args.num_classes = len(dataset.classes)
    args.imgDim = 1

    return t_loader, v_loader, dataset

## Augmentation + Normalization for full training
train_transforms = {'train': transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(args.img_scale),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandomErasing(),
    normalize_img,
]),
## Normalization only for validation and test
'valid':transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(args.img_scale),
    transforms.ToTensor(),
    normalize_img
])}

test_trans = {'test': train_transforms['valid']}

import torch.nn.functional as F
def testModel(test_dir, local_model):
    # print ('Testing model: {}'.format(str(local_model)))
    if args.use_cuda:
        local_model.cuda()
    local_model.eval()

    test_set = QuickDrawDataset(args.data_path, split='test', transform=test_trans)
    test_loader = TestQuickDrawLoader(test_set, batch_size=args.batch_size, pin_memory=(torch.cuda.is_available()), num_workers=0)

    predictions = []
    for i, (images, _) in enumerate(test_loader):
        if use_cuda:
            images = images.cuda()
            images = Variable(images)

        # compute y_pred
        y_pred = F.softmax(model(images), dim=1).data.max(1)[1].cpu().numpy()
        predictions.extend(test_set.classes[y_pred])

    return predictions

def predictions_to_csv(predictions, csv_path):
    fd = pd.DataFrame({"Categroy": predictions})
    fd.to_csv(csv_path, index_label="Id")

def make_checkpoint(model, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model.state_dict(), name + '_cnn.pth')

if __name__ == '__main__':

    # tensorboad
    # use_tensorboard = False
    use_tensorboard = False and CrayonClient is not None

    if use_tensorboard == True:
        cc = CrayonClient(hostname="localhost", port=8889)
        cc.remove_all_experiments()


    trainloader, valloader, dataset = loadDB(args)
    models = ['senet']
    for i in range (1,5):
        for m in models:
            runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fixSeed(args)
            model = selectModel(args, m)
            recorder = RecorderMeter(args.epochs)  # epoc is updated
            model_name = (type(model).__name__)

            exp_name = datetime.datetime.now().strftime(model_name + '_' + args.dataset + '_%Y-%m-%d_%H-%M-%S')
            if use_tensorboard == True:
                exp = cc.create_experiment(exp_name)

            # if model_name =='NoneType':
            #     EXIT
            mPath = args.save_path + '/' + args.dataset + '/' + model_name + '/'
            args.save_path_model = mPath
            if not os.path.isdir(args.save_path_model):
                mkdir_p(args.save_path_model)
            log = open(os.path.join(args.save_path_model, 'log_seed_{}_{}.txt'.format(args.manualSeed, runId)), 'w')
            print_log('Save path : {}'.format(args.save_path_model), log)
            print_log(state, log)
            print_log("Random Seed: {}".format(args.manualSeed), log)
            print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
            print_log("torch  version : {}".format(torch.__version__), log)
            print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
            print_log("Available models:" + str(model_names), log)
            print_log("=> Final model name '{}'".format(model_name), log)
            # print_log("=> Full model '{}'".format(model), log)
            # model = torch.nn.DataParallel(model).cuda()
            model.cuda()
            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
            print('Batch size : {}'.format(args.batch_size))

            criterion = torch.nn.CrossEntropyLoss()  # multi class
            # optimizer = torch.optim.Adam(model.parameters(), args.lr)  # L2 regularization
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

            best_losses = 100
            best_acc = 0
            for epoch in tqdm(range(args.start_epoch, args.epochs)):
                train_result, accuracy_tr = train(trainloader, model, criterion, optimizer, args)
                # evaluate on validation set
                val_result, accuracy_val = validate(valloader, model, criterion,args)

                if val_result < best_losses and accuracy_val > best_acc:
                    best_losses = val_result
                    best_acc = accuracy_val
                    pre = args.save_path_model + '/' + '/pth/'
                    fName = pre + str(accuracy_val)
                    make_checkpoint(model, pre, fName)

                recorder.update(epoch, train_result, accuracy_tr, val_result, accuracy_val)
                mPath = args.save_path_model + '/'
                if not os.path.isdir(mPath):
                    os.makedirs(mPath)
                recorder.plot_curve(os.path.join(mPath, model_name + '_' + runId + '.png'), args, model)

            predictions = testModel(args.data_path, model)
            pre = args.save_path_model + '/' + '/pth/'
            fName = pre + str(accuracy_val)
            make_checkpoint(model, pre, fName)
            csv_path = str(fName + '_submission.csv')
            predictions_to_csv(predictions, csv_path)
            print(csv_path)