# Basis of the code from https://github.com/QuantScientist/Deep-Learning-Boot-Camp/tree/master/Kaggle-PyTorch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from models import create_model
from transforms import create_transforms
from configparser import ConfigParser
from torch.autograd import Variable
import torch
import datetime
import sys
from quickdrawdataset import *

def train(train_loader, model, criterion, optimizer, model_config, pbar):
    model.cuda()
    criterion.cuda()

    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

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

        if i % model_config.getint("print_freq") == 0:
            tqdm.write('TRAIN: LOSS-->{loss.val:.4f} ({loss.avg:.4f})\t' 'ACC-->{acc.val:.3f}% ({acc.avg:.3f}%) ({i}/{len})'
                       .format(loss=losses, acc=acc, i=i, len=len(train_loader.batch_sampler)))
        pbar.refresh()
    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))

def validate(val_loader, model, criterion, args, pbar):
    model.cuda()
    criterion.cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # compute y_pred
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            # measure accuracy and record loss
            prec1, temp_var = accuracy(y_pred.data, labels.data, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))

            if i % model_config.getint("print_freq") == 0:
                tqdm.write('VAL: LOSS-->{loss.val:.4f} ({loss.avg:.4f})\t' 'ACC-->{acc.val:.3f}% ({acc.avg:.3f}%) ({i}/{len})'
                    .format(loss=losses, acc=acc, i=i, len=len(val_loader.batch_sampler)))
            pbar.refresh()
    tqdm.write('FINAL LOSS: {loss.avg:.3f}\t FINAL ACC: {acc.avg:.3f}'.format(loss=losses, acc=acc))
    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))


def loadDB(model_config, transformations):
    # Data
    tqdm.write('==> Preparing dataset')

    dataset = QuickDrawDataset(model_config.get("data_path"), split='train', transform=transformations)

    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - int(model_config.getfloat("validationRatio") * len(dataset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_indices = indices[len(indices) - int(model_config.getfloat("validationRatio") * len(dataset)):]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    t_loader = TrainingQuickDrawLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=train_sampler,
                                               pin_memory=(torch.cuda.is_available()), num_workers=4)

    v_loader = ValidationQuickDrawLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4)

    tqdm.write("Training size: {}\tValidation size: {}".format(len(train_indices), len(valid_indices)))
    tqdm.write('#Classes: {}'.format(len(classes)))

    return t_loader, v_loader, dataset

if __name__ == '__main__':
    config = ConfigParser()
    config.read('models.config')
    for model_name in config.sections():
        model_config = config[model_name]


        runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model = create_model(model_config)

        if model is None:
            tqdm.write("Model does not exist in {} mode.".format("pretrained" if model_config.getboolean("pretrained") else "non-pretrained"))
            break

        transformations = create_transforms(model_config)

        trainloader, valloader, dataset = loadDB(model_config, transformations)


        exp_name = datetime.datetime.now().strftime(model_name + '_%Y-%m-%d_%H-%M-%S')

        mPath = os.path.join(model_config.get("save_path"), "quickdraw", model_name)
        save_path_model = mPath
        if not os.path.isdir(mPath):
            os.makedirs(mPath, exist_ok=True)
        log = open(os.path.join(mPath, 'log_{}.txt'.format(runId)), 'w')
        print_log('Save path : {}'.format(mPath), log)
        print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
        print_log("torch  version : {}".format(torch.__version__), log)
        print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
        print_log("=> Final model name '{}'".format(model_name), log)
        model.cuda()
        cudnn.benchmark = True
        tqdm.write('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        tqdm.write('Batch size : {}'.format(model_config.get("batch_size")))

        criterion = torch.nn.CrossEntropyLoss()  # multi class
        if model_config.get("optimizer") == "adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=model_config.getfloat("lr"),
                                         weight_decay=model_config.getfloat("weight_decay"))
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=model_config.getfloat("lr"),
                                        momentum=model_config.getfloat("momentum"),
                                        weight_decay=model_config.getfloat("weight_decay"),
                                        nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=model_config.getint("scheduler_step_size"),
                                                    gamma=model_config.getfloat("scheduler_gamma"))

        best_losses = 100
        best_acc = 0
        n_bad_epoch = 0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        pre = os.path.join(mPath, 'pth_{}'.format(runId))
        pbar = tqdm(range(model_config.getint("start_epoch"), model_config.getint("epochs")))
        for epoch in pbar:
            scheduler.step()

            train_result, accuracy_tr = train(trainloader, model, criterion, optimizer, model_config, pbar)
            train_losses.append(train_result)
            train_accuracies.append(accuracy_tr)
            # evaluate on validation set
            val_result, accuracy_val = validate(valloader, model, criterion, model_config, pbar)
            val_losses.append(val_result)
            val_accuracies.append(accuracy_val)

            if val_result < best_losses and accuracy_val > best_acc:
                best_losses = val_result
                best_acc = accuracy_val
                fName = os.path.join(pre, str(accuracy_val) + "_" + str(val_result))
                make_checkpoint(model, pre, fName)
                n_bad_epoch = 0
            else:
                n_bad_epoch += 1

            if n_bad_epoch >= model_config.getint("bad_epoch_threshold"):
                # tqdm.write(model_config.get("bad_epoch_threshold"), "epochs since last best validation. Stopping.")
                break

        metrics_path = os.path.join(pre, "metrics.csv")
        metrics = np.array([train_losses, train_accuracies, val_losses, val_accuracies])
        np.savetxt(metrics_path, metrics, delimiter=',')

        metrics_path = os.path.join(pre, "metrics_test.csv")
        metrics = np.array([train_losses, train_accuracies, val_losses, val_accuracies]).T
        np.savetxt(metrics_path, metrics, delimiter=',')