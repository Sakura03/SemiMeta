import os, argparse, torch, math, time, random
from os.path import join, isfile
import torch.nn.functional as F
from torch.optim import SGD
from torch.distributions import Beta
from tensorboardX import SummaryWriter

from dataloader import dataloader
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint
from model import ConvLarge, shakeshake26, wideresnet28

parser = argparse.ArgumentParser()
# Basic configuration
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data-path', type=str, default='./data', help='Data path')
parser.add_argument('--num-label', type=int, default=4000)
parser.add_argument('-a', '--architecture', type=str, default='convlarge', choices=['convlarge', 'shakeshake', 'wrn'], help='Network architecture')
parser.add_argument('--mix-up', action='store_true', help='Use mix-up augmentation')
parser.add_argument('--alpha', type=float, default=1., help='Concentration parameter of Beta distribution')
# Training setting
parser.add_argument('--total-steps', type=int, default=400000, help='Start step (for resume)')
parser.add_argument('--start-step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=0.1, help='Maximum learning rate')
parser.add_argument('--warmup', type=int, default=4000, help='Warmup iterations')
parser.add_argument('--const-steps', type=int, default=0, help='Number of iterations of constant lr')
parser.add_argument('--lr-decay', type=str, default='step', choices=['step', 'linear', 'cosine'], help='Learning rate annealing strategy')
parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate annealing multiplier')
parser.add_argument('--milestones', type=eval, default=[300000, 350000], help='Learning rate annealing steps')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
parser.add_argument('--resume', type=str, default=None, help='Resume model from a checkpoint')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
parser.add_argument('--print-freq', type=int, default=100, help='Print and log frequency')
parser.add_argument('--test-freq', type=int, default=400, help='Test frequency')
parser.add_argument('--save-path', type=str, default='./results/tmp', help='Save path')
args = parser.parse_args()
args.num_classes = 100 if args.dataset == 'cifar100' else 10

# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create directories if not exist
make_folder(args.save_path)
logger = Logger(join(args.save_path, 'log.txt'))
writer = SummaryWriter(log_dir=args.save_path)
logger.info('Called with args:')
logger.info(args)
torch.backends.cudnn.benchmark = True

# Define dataloader
logger.info("Loading data...")
train_loader, test_loader = dataloader(
        dset = args.dataset,
        path = args.data_path,
        bs = args.batch_size,
        num_workers = args.num_workers,
        num_labels = args.num_label,
        num_iters = args.total_steps,
        return_unlabel = args.mix_up,
        save_path = args.save_path
        )

# Build model and optimizer
logger.info("Building model and optimizer...")
if args.architecture == "convlarge":
    model = ConvLarge(num_classes=args.num_classes, stochastic=True).cuda()
elif args.architecture == "shakeshake":
    model = shakeshake26(num_classes=args.num_classes).cuda()
elif args.architecture == "wrn":
    model = wideresnet28(num_classes=args.num_classes).cuda()
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
logger.info("Model:\n%s\nOptimizer:\n%s" % (str(model), str(optimizer)))

# Optionally build beta distribution
if args.mix_up:
    beta_distribution = Beta(torch.tensor([args.alpha]), torch.tensor([args.alpha]))

# Optionally resume from a checkpoint
if args.resume is not None:
    if isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_step = checkpoint['step']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (step {})".format(args.resume, checkpoint['step']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

def compute_lr(step):
    if step < args.warmup:
        lr = args.lr * step / args.warmup
    elif step < args.warmup + args.const_steps:
        lr = args.lr
    elif args.lr_decay == "step":
        lr = args.lr
        for milestone in args.milestones:
            if step > milestone:
                lr *= args.gamma
    elif args.lr_decay == "linear":
        lr = args.lr * ( 1. - (step-args.warmup-args.const_steps) / (args.total_steps-args.warmup-args.const_steps) )
    elif args.lr_decay == "cosine":
        lr = args.lr * ( 1. + math.cos( (step-args.warmup-args.const_steps) / (args.total_steps-args.warmup-args.const_steps) *  math.pi ) ) / 2.
    return lr

def main():
    data_times, batch_times, losses, acc = [AverageMeter() for _ in range(4)]
    if args.mix_up:
        unlabel_acc = AverageMeter()
    best_acc = 0.
    model.train()
    logger.info("Start training...")
    for step in range(args.start_step, args.total_steps):
        # Load data and distribute to devices
        data_start = time.time()
        if args.mix_up:
            label_img, label_gt, unlabel_img, unlabel_gt = next(train_loader)
        else:
            label_img, label_gt = next(train_loader)
        
        label_img = label_img.cuda()
        label_gt = label_gt.cuda()
        if args.mix_up:
            unlabel_img = unlabel_img.cuda()
            unlabel_gt = unlabel_gt.cuda()
            _label_gt = F.one_hot(label_gt, num_classes=args.num_classes).float()
        data_end = time.time()

        # Compute learning rate
        lr = compute_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if args.mix_up:
            # Adopt mix-up augmentation
            model.eval()
            with torch.no_grad():
                label_pred = model(label_img)
                unlabel_pred = F.softmax(model(unlabel_img), dim=1) 
                alpha = beta_distribution.sample((args.batch_size,)).cuda()
                _alpha = alpha.view(-1, 1, 1, 1)
                interp_img = (label_img * _alpha + unlabel_img * (1. - _alpha)).detach()
                interp_pseudo_gt = (_label_gt * alpha + unlabel_pred * (1. - alpha)).detach()
            model.train()
            interp_pred = model(interp_img)
            loss = F.kl_div(F.log_softmax(interp_pred, dim=1), interp_pseudo_gt, reduction='batchmean')
        else:
            # Regular label loss
            label_pred = model(label_img)
            loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
        
        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        top1, = accuracy(label_pred, label_gt, topk=(1,))
        if args.mix_up:
            unlabel_top1, = accuracy(unlabel_pred, unlabel_gt, topk=(1,))
        # Update AverageMeter stats
        data_times.update(data_end - data_start)
        batch_times.update(time.time() - data_end)
        losses.update(loss.item(), label_img.size(0))
        acc.update(top1.item(), label_img.size(0))
        if args.mix_up:
            unlabel_acc.update(unlabel_top1.item(), unlabel_img.size(0))
        
        # Print and log
        if step % args.print_freq == 0:
            logger.info("Step: [{0:05d}/{1:05d}] Dtime: {dtimes.avg:.3f} Btime: {btimes.avg:.3f} "
                        "loss: {losses.val:.3f} (avg {losses.avg:.3f}) Lacc: {label.val:.3f} (avg {label.avg:.3f}) "
                        "LR: {2:.4f}".format(step, args.total_steps, optimizer.param_groups[0]['lr'],
                                             dtimes=data_times, btimes=batch_times, losses=losses, label=acc))

        # Test and save model
        if (step + 1) % args.test_freq == 0 or step == args.total_steps - 1:
            val_acc = evaluate(test_loader, model)
            
            # Remember best accuracy and save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            logger.info("Best Accuracy: %.5f" % best_acc)
            save_checkpoint({
                'step': step + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.save_path, filename="checkpoint.pth")
            
            # Write to tfboard
            writer.add_scalar('train/label-acc', top1.item(), step)
            if args.mix_up:
                writer.add_scalar('train/unlabel-acc', unlabel_top1.item(), step)
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('test/accuracy', val_acc, step)
            
            # Reset AverageMeters
            losses.reset()
            acc.reset()
            if args.mix_up:
                unlabel_acc.reset()

@torch.no_grad()
def evaluate(test_loader, model):
    batch_time, losses, acc = [AverageMeter() for _ in range(3)]
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (data, target) in enumerate(test_loader):
        # Load data
        data = data.cuda()
        target = target.cuda()
        
        # Compute output
        pred = model(data)
        loss = F.cross_entropy(pred, target, reduction='mean')
        
        # Measure accuracy and record loss
        top1, = accuracy(pred, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        acc.update(top1.item(), data.size(0))
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}] Time {btime.val:.3f} (avg={btime.avg:.3f}) '
                        'Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                        'Acc {acc.val:.3f} (avg={acc.avg:.3f})' \
                        .format(i, len(test_loader), btime=batch_time, loss=losses, acc=acc))
    
    logger.info(' * Accuracy {acc.avg:.5f}'.format(acc=acc))
    model.train()
    return acc.avg

if __name__ == "__main__":
    # Train and evaluate the model
    main()
    writer.close()
    logger.close()

