import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

import torchattacks

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_top_sim(sim_matrix):
    k = 20 # use 20 neighbor
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def test_time_tuning(model, inputs, optimizer, scaler, args):
    
    selected_idx = None
    for j in range(args.tta_steps):
        if True:
            output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = entropy_avg(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    args.alpha = args.eps / 4.0
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets, 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps))

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # model
    dset = args.test_sets
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            # 'RN50-eps1': 'pretrain/tecoa/rn50_eps1.pth.tar',
            'RN50-eps1': '/data/shenglijun/code/my_code/project/clip_adversarial_training/APT-main/apt/backbone/rn50_eps1.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        print('load robust vision encoder')

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
                param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    results = {}
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 1

        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(dset))
        
        results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform)
        del val_dataset, val_loader
        if args.eps <= 0:
            print_log = "=> Acc. on testset [{}]: Clean Acc @1 {}/ TTA Clean Acc @1 {}".format(dset, results[0], results[1])
            save_log = {'clean_acc': results[0], 'tta_clean_acc': results[1]}
        else:
            print_log = "=> Acc. on testset [{}]: Adv Acc @1 {}/ TTA Adv Acc @1 {} ".format(dset, results[0], results[1])
            save_log = {'adv_acc': results[0], 'tta_adv_acc': results[1]}
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        if args.eps > 0.0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv_image = atk(image, target)        
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        
        images = torch.cat(images, dim=0)

        # reset model
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            clip_output = model(image)
            clip_features, _, _ = model.forward_features(images)
            clip_outputs = model(images)

        assert args.tta_steps > 0
        test_time_tuning(model, images, optimizer, scaler, args)
        with torch.no_grad():
            tuned_outputs = model(images)
        
        sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        score = get_top_sim(sim_matrix_images)
        weight = torch.nn.functional.softmax(score/0.01, dim=-1)
        tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))
       
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, clip_acc1={}, tta_acc1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            else:
                print_log = 'iter:{}/{}, clip_adv1={}, tta_adv1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    progress.display_summary()

    return [top1.avg, tpt1.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps_temp/temp')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])

    main()
