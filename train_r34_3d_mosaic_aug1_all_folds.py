from torch.cuda.amp import autocast, GradScaler
import time, gc
import random
import glob
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from resnet3d import generate_model
#from torchmetrics.classification import BinaryFBetaScore


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    debug = False #
    comp_dataset_path = 'crop256_16_32'
    # resnet34d kaggle file
    exp_name = 'train_r34_3d_mosaic'
    desc = '  crop256_16_32,stride=128, mosaic p=0.5, r34 bb'
    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Resnet34_3d'
    Z_START = 16
    Z_DIMS = 32


    # ============== training cfg =============
    size = 256
    train_batch_size = 16
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 50
    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = '1'
    all_ids = ['1', '2a', '2b', '3']
    metric_direction = 'maximize'  # maximize, 'minimize'

    # ============== fixed =============

    inf_weight = 'best'
    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000
    print_freq = 50
    num_workers = 4
    seed = 42

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'outputs/{comp_name}/{exp_name}/'

    model_dir = outputs_path + \
                f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                           border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.Normalize(
            mean=[0] * Z_DIMS,
            std=[1] * Z_DIMS
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * Z_DIMS,
            std=[1] * Z_DIMS
        ),
        ToTensorV2(transpose_mask=True),
    ]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger



def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug

def load_image_and_mask(self, idx):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img_fn = self.img_fns[idx]
        masks_fn = self.masks_fns[idx]
        img = np.load(img_fn)
        mask = cv2.imread(masks_fn, 0).astype("float32")
        mask[mask > 0] = 1.
        yx = masks_fn.split('/')[-1].split('.')[0]
        return img, mask, yx


def load_mosaic(self, index):
    # loads images in a mosaic

    # labels4 = []
    s = CFG.size
    h = CFG.size
    w = CFG.size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.img_fns) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, mask, yx = load_image_and_mask(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 0, dtype=np.uint16)  # base image with 4 tiles
            mask4 = np.full((s * 2, s * 2), 0.) #, dtype=np.uint8
            # mask4=mask
            yx4 = yx
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

    return img4, mask4, yx4

class CustomDataset(Dataset):
    def __init__(self, cfg, labels=None, transform=None, mode='train'):
        if mode == 'train':
            folds = list(set(CFG.all_ids) - set([CFG.valid_id]))
        else:
            folds = [CFG.valid_id]
        print('folds:', folds)
        img_fns = []
        masks_fns = []
        for fold in folds:
            img_fns += glob.glob(f'{CFG.comp_dataset_path}/{fold}/*.npy')
            img_fns = sorted(img_fns)
            masks_fns += glob.glob(f'{CFG.comp_dataset_path}/{fold}/*.png')
            masks_fns = sorted(masks_fns)
        if CFG.debug:
            img_fns = img_fns[:100]
            masks_fns = masks_fns[:100]
            print('DEBUG!!!!')
        print('!!!', len(img_fns), len(masks_fns))
        self.img_fns = img_fns
        self.masks_fns = masks_fns
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        if  self.mode=='train' and random.randint(0,1):
            img,mask,yx = load_mosaic(self,idx)
        else:
            img,mask,yx =load_image_and_mask(self, idx)
        _, y, x = yx.split('_')
        y = int(y)
        x = int(x)
        if self.transform:
            data = self.transform(image=img, mask=mask)
            img = data['image']
            mask = data['mask']

        return img.unsqueeze(0), mask, y, x


#ds= CustomDataset(CFG,transform=get_transforms(data='train', cfg=CFG)) #i,m,y,x=ds[0]
#ds= CustomDataset(CFG, transform=get_transforms(data='valid', cfg=CFG), mode='valid')
# ba=next(iter(train_loader))
# i,m,y,x=ba
from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
#fbeta = BinaryFBetaScore(beta=0.5).cuda()

alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(
    mode='binary', log_loss=False, alpha=alpha, beta=beta)


def criterion(y_pred, y_true):
     return BCELoss(y_pred, y_true)



class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = generate_model(model_depth=34, n_input_channels=1)
        self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels, y, x) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device,dtype=torch.float)
        labels = labels.to(device,dtype=torch.float)
        batch_size = labels.size(0)
        with autocast(CFG.use_amp):
            y_preds = model(images).squeeze(1)
            # print('y_preds',y_preds.shape,'labels:',labels.shape)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    del y_preds
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg


def valid_fn(valid_loader, model, criterion, device, valid_mask_gt):  # valid_xyxys
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels, y, x) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        y1 = y.numpy()
        x1 = x.numpy()
        y2 = y1 + CFG.size
        x2 = x1 + CFG.size
        # print(y1,y2,x1,x2)
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images).squeeze(1)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        for i in range(len(y_preds)):
            mask_pred[y1[i]:y2[i], x1[i]:x2[i]] += y_preds[i]  # .squeeze(0)
            mask_count[y1[i]:y2[i], x1[i]:x2[i]] += np.ones((CFG.size, CFG.size))
    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    del y_preds
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg, mask_pred




def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(30, 70 + 1, 5)) / 100:  # np.array(range(10, 50 + 1, 5)) / 100:

        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th

    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th


if __name__ == '__main__':
    cfg_init(CFG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Logger = init_logger(log_file=CFG.log_path)
    for fold in CFG.all_ids:
        CFG.valid_id = fold
        Logger.info(f'\n\n-------- exp_info fr{CFG.valid_id}-----------------')
        Logger.info(f'{CFG.desc}')
        valid_mask_gt = cv2.imread(f"data/train/{CFG.valid_id}/inklabels.png", 0)
        valid_mask_gt[valid_mask_gt > 0] = 1.
        pad0 = (CFG.size - valid_mask_gt.shape[0] % CFG.size)
        pad1 = (CFG.size - valid_mask_gt.shape[1] % CFG.size)
        valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
        train_dataset = CustomDataset(CFG, transform=get_transforms(data='train', cfg=CFG), mode='train')
        valid_dataset = CustomDataset(CFG, transform=get_transforms(data='valid', cfg=CFG), mode='valid')
        print('train_dataset:', len(train_dataset), 'valid_dataset:', len(valid_dataset))
        train_loader = DataLoader(train_dataset,
                                  batch_size=CFG.train_batch_size,
                                  shuffle=True,
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                  )
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=CFG.valid_batch_size,
                                  shuffle=False,
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

        model = SegModel()
        model.load_pretrained_weights(torch.load("r3d34_K_200ep.pth")["state_dict"])
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=CFG.lr)
        scheduler = get_scheduler(CFG, optimizer)


        if CFG.metric_direction == 'minimize':
            best_score = np.inf
        elif CFG.metric_direction == 'maximize':
            best_score = -1

        best_loss = np.inf

        for epoch in range(CFG.epochs):

            start_time = time.time()

            # train
            avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

            # eval
            avg_val_loss, mask_pred = valid_fn(
                valid_loader, model, criterion, device, valid_mask_gt)  # valid_xyxys

            scheduler_step(scheduler, avg_val_loss, epoch)

            best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

            # score = avg_val_loss
            score = best_dice

            elapsed = time.time() - start_time

            Logger.info(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
            Logger.info(
                f'Epoch {epoch + 1} - avgScore: {score:.4f}')

            if CFG.metric_direction == 'minimize':
                update_best = score < best_score
            elif CFG.metric_direction == 'maximize':
                update_best = score > best_score

            if update_best:
                # best_loss = avg_val_loss
                best_score = score

                Logger.info(
                    f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')

                torch.save({'model': model.state_dict()},
                           # 'preds': mask_pred},
                           CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')


        del model,train_dataset,valid_dataset,train_loader,valid_loader,valid_mask_gt
        gc.collect()
        torch.cuda.empty_cache()

