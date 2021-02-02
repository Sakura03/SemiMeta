import os, argparse, time, warnings, torch, torchvision
import numpy as np
from os.path import join, isfile
from utils import Logger, save_checkpoint, AverageMeter, accuracy, CosAnnealingLR
from tensorboardX import SummaryWriter
# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
# distributed
import torch.nn.functional as F
from torch.distributions import Beta
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.simplefilter('error')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('-a', '--arch', default='resnet18', type=str, metavar='STR', help='model architecture')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')
parser.add_argument('--num-classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.1, help="learning rate")
parser.add_argument('--epochs', type=int, default=120, help="epochs")
parser.add_argument('--wd', '--weight-decay', type=float, default=1e-4, help="weight decay")
parser.add_argument('--imsize', default=256, type=int, metavar='N', help='image resize size')
parser.add_argument('--imcrop', default=224, type=int, metavar='N', help='image crop size')
parser.add_argument('--warmup', default=5, type=int, help="warmup epochs (small lr and do not impose sparsity)")
parser.add_argument('--gamma', default=0.1, type=float, metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--alpha', type=float, default=1., help='Concentration parameter of Beta distribution')
parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for gradient estimation')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--tmp', default="results/tmp", type=str, help='tmp folder')
# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dali-cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
args = parser.parse_args()

def data_gen(loader):
    while True:
        for data in loader:
            yield data
        loader.reset()

# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, rec_name, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # MXnet rec reader
        self.input = ops.MXNetReader(path=join(data_dir, "%s.rec"%rec_name), index_path=join(data_dir, "%s.idx"%rec_name),
                                     random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path=join(data_dir, "val.rec"), index_path=join(data_dir, "val.idx"),
                                     random_shuffle=False, shard_id=args.local_rank, num_shards=args.world_size)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]  

torch.backends.cudnn.benchmark = True
os.makedirs(args.tmp, exist_ok=True)
if args.local_rank == 0:
    tfboard_writer = SummaryWriter(log_dir=args.tmp)
    logger = Logger(join(args.tmp, "log.txt"))
# Build Beta distribution
beta_distribution = Beta(torch.tensor([args.alpha]), torch.tensor([args.alpha]))

def main():
    if args.local_rank == 0:
        logger.info(args)
    
    # Pytorch distributed setup
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    
    # Build DALI dataloader
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers,
                           device_id=args.local_rank, data_dir=args.data, 
                           rec_name="train_label", crop=args.imcrop, dali_cpu=args.dali_cpu)
    pipe.build()
    label_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    label_provider = data_gen(label_loader)

    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers,
                           device_id=args.local_rank, data_dir=args.data, 
                           rec_name="train_unlabel", crop=args.imcrop, dali_cpu=args.dali_cpu)
    pipe.build()
    unlabel_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
    pipe = HybridValPipe(batch_size=50, num_threads=args.workers,
                         device_id=args.local_rank, data_dir=args.data,
                         crop=args.imcrop, size=args.imsize)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    unlabel_loader_len = int(np.ceil(unlabel_loader._size/args.batch_size))
    
    # Define model and optimizer
    model_name = "torchvision.models.%s(num_classes=%d)" % (args.arch, args.num_classes)
    model = eval(model_name).cuda()
    if args.local_rank == 0:
        logger.info("Model details:")
        logger.info(model)
    model = DDP(model, delay_allreduce=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    if args.local_rank == 0:
        logger.info("Optimizer details:")
        logger.info(optimizer)
    
    # Optionally resume from a checkpoint
    if args.resume is not None:
        assert isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        if args.local_rank == 0:
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        if args.local_rank == 0: 
            best_acc1 = checkpoint['best_acc1']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))    
    else:
        args.start_epoch = 0
        if args.local_rank == 0: best_acc1 = 0.
    
    # Define learning rate scheduler
    scheduler = CosAnnealingLR(loader_len=unlabel_loader_len, epochs=args.epochs,
                               lr_max=args.lr, warmup_epochs=args.warmup,
                               last_epoch=args.start_epoch-1)
    
    for epoch in range(args.start_epoch, args.epochs):
        # Train and evaluate
        train(label_provider, unlabel_loader, model, optimizer, scheduler, epoch)
        acc1, acc5 = validate(val_loader, model)
        
        if args.local_rank == 0:
            # Write to tfboard
            tfboard_writer.add_scalar('test/acc1', acc1, epoch)
            tfboard_writer.add_scalar('test/acc5', acc5, epoch)
            
            # Remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
            logger.info("Best acc1=%.5f" % best_acc1)
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.tmp, filename="checkpoint.pth")

def train(label_provider, unlabel_loader, model, optimizer, scheduler, epoch):
    if args.local_rank == 0:
        data_times, batch_times, label_losses, unlabel_losses, label_acc1, label_acc5, unlabel_acc1, unlabel_acc5 = \
                [AverageMeter() for _ in range(8)]
        unlabel_loader_len = int(np.ceil(unlabel_loader._size/args.batch_size))
    
    # Switch to train mode
    model.train()
    if args.local_rank == 0:
        end = time.time()
    for i, data in enumerate(unlabel_loader):
        # Load data and distribute to devices
        label_data = next(label_provider)
        label_img = label_data[0]["data"]
        label_gt = label_data[0]["label"].squeeze().cuda().long()
        _label_gt = F.one_hot(label_gt, num_classes=args.num_classes).float()
        unlabel_img = data[0]["data"]
        unlabel_gt = data[0]["label"].squeeze().cuda().long()
        if args.local_rank == 0:
            start = time.time()
        
        # Compute the learning rate
        lr = scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
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
            unlabel_grad = F.softmax(unlabel_pred_pos, dim=1) - F.softmax(unlabel_pred_neg, dim=1)
            unlabel_grad.div_(epsilon)
            
            # Update and normalize pseudo-labels
            unlabel_pseudo_gt.sub_(lr, unlabel_grad)
            torch.relu_(unlabel_pseudo_gt)
            sums = torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
            unlabel_pseudo_gt /= torch.where(sums == 0., torch.ones_like(sums), sums)

        # Training mode
        model.train()
        
        # First compute label loss (mix-up augmentation)
        with torch.no_grad():
            alpha = beta_distribution.sample((args.batch_size,)).cuda()
            _alpha = alpha.view(-1, 1, 1, 1)
            interp_img = (label_img * _alpha + unlabel_img * (1. - _alpha)).detach()
            interp_pseudo_gt = (_label_gt * alpha + unlabel_pseudo_gt * (1. - alpha)).detach()
        interp_pred = model(interp_img)
        label_loss = F.kl_div(F.log_softmax(interp_pred, dim=1), interp_pseudo_gt, reduction='batchmean')
            
        # Then compute unlabel loss with `unlabel_pseudo_gt`
        unlabel_pred = model(unlabel_img)
        unlabel_loss = torch.norm(F.softmax(unlabel_pred, dim=1)-unlabel_pseudo_gt, p=2, dim=1).pow(2).mean()
        loss = label_loss + unlabel_loss
        
        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        label_top1, label_top5 = accuracy(label_pred, label_gt, topk=(1, 5))
        unlabel_top1, unlabel_top5 = accuracy(unlabel_pred, unlabel_gt, topk=(1, 5))
        
        # Gather tensors from different devices
        label_loss = reduce_tensor(label_loss)
        unlabel_loss = reduce_tensor(unlabel_loss)
        label_top1 = reduce_tensor(label_top1)
        label_top5 = reduce_tensor(label_top5)
        unlabel_top1 = reduce_tensor(unlabel_top1)
        unlabel_top5 = reduce_tensor(unlabel_top5)
        
        # Update AverageMeter stats
        if args.local_rank == 0:
            data_times.update(start - end)
            batch_times.update(time.time() - start)
            label_losses.update(label_loss.item(), label_img.size(0))
            unlabel_losses.update(unlabel_loss.item(), unlabel_img.size(0))
            label_acc1.update(label_top1.item(), label_img.size(0))
            label_acc5.update(label_top5.item(), label_img.size(0))
            unlabel_acc1.update(unlabel_top1.item(), unlabel_img.size(0))
            unlabel_acc5.update(unlabel_top5.item(), unlabel_img.size(0))
        # torch.cuda.synchronize()
        
        # Log training info
        if i % args.print_freq == 0 and args.local_rank == 0:
            tfboard_writer.add_scalar("train/learning-rate", lr, epoch*unlabel_loader_len+i)
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.avg:.3f} Dt {data_time.avg:.3f} '
                        'Lloss: {llosses.val:.3f} (avg {llosses.avg:.3f}) Uloss: {ulosses.val:.3f} (avg {ulosses.avg:.3f}) '
                        'Lacc1: {label1.val:.3f} (avg {label1.avg:.3f}) Lacc5: {label5.val:.3f} (avg {label5.avg:.3f}) '
                        'Uacc1: {unlabel1.val:.3f} (avg {unlabel1.avg:.3f}) Uacc5: {unlabel5.val:.3f} (avg {unlabel5.avg:.3f}) '
                        'LR: {4:.3E} '.format(
                                epoch, args.epochs, i, unlabel_loader_len, lr, batch_time=batch_times, data_time=data_times,
                                llosses=label_losses, ulosses=unlabel_losses, label1=label_acc1, label5=label_acc5,
                                unlabel1=unlabel_acc1, unlabel5=unlabel_acc5
                                ))
        if args.local_rank == 0:
            end = time.time()
    
    # Reset the unlabel loader
    unlabel_loader.reset()
    if args.local_rank == 0:
        tfboard_writer.add_scalar('train/label-loss', label_losses.avg, epoch)
        tfboard_writer.add_scalar('train/unlabel-loss', unlabel_losses.avg, epoch)
        tfboard_writer.add_scalar('train/label-acc1', label_acc1.avg, epoch)
        tfboard_writer.add_scalar('train/label-acc5', label_acc5.avg, epoch)
        tfboard_writer.add_scalar('train/unlabel-acc1', unlabel_acc1.avg, epoch)
        tfboard_writer.add_scalar('train/unlabel-acc5', unlabel_acc5.avg, epoch)

@torch.no_grad()
def validate(val_loader, model):
    if args.local_rank == 0:
        losses, top1, top5 = [AverageMeter() for _ in range(3)]
        val_loader_len = int(np.ceil(val_loader._size/50))
    
    # Switch to evaluate mode
    model.eval()
    for i, data in enumerate(val_loader):
        image = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        
        # Compute output
        prediction = model(image)
        loss = F.cross_entropy(prediction, target, reduction='mean')
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(prediction, target, topk=(1, 5))
        
        # Gather tensors from different devices
        loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        
        # Update meters and log info
        if args.local_rank == 0:
            losses.update(loss.item(), image.size(0))
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                            'Acc1 {top1.val:.3f} (avg={top1.avg:.3f}) Acc5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                            .format(i, val_loader_len, loss=losses, top1=top1, top5=top5))
    if args.local_rank == 0:
        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))
    
    # Reset the validation loader
    val_loader.reset()
    top1 = torch.tensor([top1.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    top5 = torch.tensor([top5.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    dist.broadcast(top1, 0)
    dist.broadcast(top5, 0)
    return top1.cpu().item(), top5.cpu().item()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.reduce(rt, 0, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
    if args.local_rank == 0:
        tfboard_writer.close()
        logger.close()
