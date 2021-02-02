import os, argparse, torch, math, time, random
from os.path import join, isfile
import torch.nn.functional as F
from torch.optim import SGD
from torch.distributions import Beta
from tensorboardX import SummaryWriter

from dataloader import dataloader
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint, compute_weight
from model import ConvLarge, shakeshake26, wideresnet28

parser = argparse.ArgumentParser()
# Basic configuration
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data-path', type=str, default='./data', help='Data path')
parser.add_argument('--num-label', type=int, default=4000, help='Number of labeled data')
parser.add_argument('--additional', type=str, default='None', choices=['None', '500k', '237k'], help='Additional unlabeled data from TinyImages Dataset')
parser.add_argument('-a', '--architecture', type=str, default='convlarge', choices=['convlarge', 'shakeshake', 'wrn'], help='Network architecture')
parser.add_argument('--mix-up', action='store_true', help='Use mix-up augmentation')
parser.add_argument('--alpha', type=float, default=1., help='Concentration parameter of Beta distribution')
parser.add_argument('--weight', type=float, default=1., help='re-weighting scalar for the additional loss')
parser.add_argument('--consistency', type=str, default='mse', choices=['kl', 'mse'], help='Consistency loss type')
# Training setting
parser.add_argument('--total-steps', type=int, default=400000, help='Start step (for resume)')
parser.add_argument('--start-step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for gradient estimation')
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
        return_unlabel = True,
        additional = args.additional,
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
    data_times, batch_times, label_losses, unlabel_losses, label_acc, unlabel_acc = [AverageMeter() for _ in range(6)]
    if args.mix_up: interp_losses = AverageMeter()
    best_acc = 0.
    logger.info("Start training...")
    for step in range(args.start_step, args.total_steps):
        # Load data and distribute to devices
        data_start = time.time()
        label_img, label_gt, unlabel_img, unlabel_gt = next(train_loader)
        
        label_img = label_img.cuda()
        label_gt = label_gt.cuda()
        unlabel_img = unlabel_img.cuda()
        unlabel_gt = unlabel_gt.cuda()
        _label_gt = F.one_hot(label_gt, num_classes=args.num_classes).float()
        data_end = time.time()
        
        # Compute learning rate and meta learning rate
        lr = compute_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        weight = compute_weight(args.weight, step, args.warmup)

        ### First-order Approximation ###
        _concat = lambda xs: torch.cat([x.view(-1) for x in xs])
        # Evaluation mode
        model.eval()
        # Forward label data and perform backward pass
        label_pred = model(label_img)
        label_loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
        dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
        
        with torch.no_grad():
            # Compute the unlabel pseudo-gt
            unlabel_pred = model(unlabel_img)
            unlabel_pseudo_gt = F.softmax(unlabel_pred, dim=1)
            # Compute step size for first-order approximation
            epsilon = args.epsilon / torch.norm(_concat(dtheta))
            
            # Forward finite difference
            for p, g in zip(model.parameters(), dtheta):
                p.data.add_(epsilon, g)            
            unlabel_pred_pos = model(unlabel_img)
            # Backward finite difference
            for p, g in zip(model.parameters(), dtheta):
                p.data.sub_(2.*epsilon, g)
            unlabel_pred_neg = model(unlabel_img)
            # Resume original params
            for p, g in zip(model.parameters(), dtheta):
                p.data.add_(epsilon, g)
            
            # Compute (approximated) gradients w.r.t pseudo-gt of unlabel data
            if args.consistency == 'kl':
                # If the consistency loss is KL Divergence
                unlabel_grad = F.log_softmax(unlabel_pred_pos, dim=1) - F.log_softmax(unlabel_pred_neg, dim=1)
                unlabel_grad.div_(2.*epsilon)
            elif args.consistency == 'mse':
                # If the consistency loss is MSE
                unlabel_grad = F.softmax(unlabel_pred_pos, dim=1) - F.softmax(unlabel_pred_neg, dim=1)
                unlabel_grad.div_(epsilon)
            
            # Update and normalize pseudo-labels
            unlabel_pseudo_gt.sub_(lr, unlabel_grad)
            torch.relu_(unlabel_pseudo_gt)
            sums = torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
            unlabel_pseudo_gt /= torch.where(sums == 0., torch.ones_like(sums), sums)
        
        # Training mode
        model.train()
        # First compute label loss
        if args.mix_up:
            # Adopt mix-up augmentation
            with torch.no_grad():
                alpha = beta_distribution.sample((args.batch_size,)).cuda()
                _alpha = alpha.view(-1, 1, 1, 1)
                interp_img = (label_img * _alpha + unlabel_img * (1. - _alpha)).detach()
                interp_pseudo_gt = (_label_gt * alpha + unlabel_pseudo_gt * (1. - alpha)).detach()
            interp_pred = model(interp_img)
            interp_loss = F.kl_div(F.log_softmax(interp_pred, dim=1), interp_pseudo_gt, reduction='batchmean')
        else:
            # Regular label loss
            label_pred = model(label_img)
            label_loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
            
        # Then compute unlabel loss with `unlabel_pseudo_gt`
        unlabel_pred = model(unlabel_img)
        if args.consistency == 'kl':
            unlabel_loss = F.kl_div(F.log_softmax(unlabel_pred, dim=1), unlabel_pseudo_gt, reduction='batchmean')
        elif args.consistency == 'mse':
            unlabel_loss = torch.norm(F.softmax(unlabel_pred, dim=1)-unlabel_pseudo_gt, p=2, dim=1).pow(2).mean()
        loss = interp_loss + weight * unlabel_loss if args.mix_up else label_loss + weight * unlabel_loss
                
        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        label_top1, = accuracy(label_pred, label_gt, topk=(1,))
        unlabel_top1, = accuracy(unlabel_pred, unlabel_gt, topk=(1,))
        # Update AverageMeter stats
        data_times.update(data_end - data_start)
        batch_times.update(time.time() - data_end)
        label_losses.update(label_loss.item(), label_img.size(0))
        unlabel_losses.update(unlabel_loss.item(), unlabel_img.size(0))
        label_acc.update(label_top1.item(), label_img.size(0))
        unlabel_acc.update(unlabel_top1.item(), unlabel_img.size(0))
        if args.mix_up: interp_losses.update(interp_loss.item(), label_img.size(0))
        
        # Print and log
        if step % args.print_freq == 0:
            logger.info("Step {0:05d} Dtime: {dtimes.avg:.3f} Btime: {btimes.avg:.3f} "
                        "Lloss: {llosses.val:.3f} (avg {llosses.avg:.3f}) Uloss: {ulosses.val:.3f} (avg {ulosses.avg:.3f}) "
                        "Lacc: {label.val:.3f} (avg {label.avg:.3f}) Uacc: {unlabel.val:.3f} (avg {unlabel.avg:.3f}) "
                        "LR: {1:.4f}".format(step, lr, dtimes=data_times, btimes=batch_times, llosses=label_losses,
                                             ulosses=unlabel_losses, label=label_acc, unlabel=unlabel_acc))
        
        # Test and save model
        if (step + 1) % args.test_freq == 0 or step == args.total_steps - 1:
            acc = evaluate(test_loader, model)
            
            # Remember best accuracy and save checkpoint
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
            logger.info("Best Accuracy: %.5f" % best_acc)
            save_checkpoint({
                'step': step + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.save_path, filename="checkpoint.pth")
            
            # Write to the tfboard
            writer.add_scalar('train/label-acc', label_acc.avg, step)
            writer.add_scalar('train/unlabel-acc', unlabel_acc.avg, step)
            writer.add_scalar('train/label-loss', label_losses.avg, step)
            writer.add_scalar('train/unlabel-loss', unlabel_losses.avg, step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('test/accuracy', acc, step)
            if args.mix_up: writer.add_scalar('train/interp-loss', interp_losses.avg, step)
            
            # Reset AverageMeters
            label_losses.reset()
            unlabel_losses.reset()
            label_acc.reset()
            unlabel_acc.reset()
            if args.mix_up: interp_losses.reset()

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
    return acc.avg

if __name__ == "__main__":
    # Train and evaluate the model
    main()
    writer.close()
    logger.close()
