import torch
import time
from torch.nn import parameter
from torch.utils.data import DataLoader
from models.models import MultiBoxLoss, tiny_detector
from utils.datasets import PasCalDataset 
from utils.utils import *
import torch.backends.cudnn as cudnn


# 定义超参数 常量
epoch = 60
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
batch_size = 32
print_freq = 30
n_classes = len(label_map)
workers = 4
step_size = 30 #stepsize of adjusting lr
scale = 0.1
resume = True
data_folder = 'data/VOC2012/json'
checkpoint_path = 'checkpoint.pth.tar'
start_epoch = 0

def train(epoch, train_loader, model, optimizer, criterion):
    model.train()

    loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter() #forawrd prop + backward prop

    start = time.time()
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        start = time.time()

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        pred_locs, pred_scores = model(images)
        optimizer.zero_grad()

        l = criterion(pred_locs, pred_scores, boxes, labels) #返回一个scalar 数字
        l.backward()
        optimizer.step()

        loss.update(l.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch [{epoch}][{i+1} / {len(train_loader)}] \t Loading time:{data_time.val:.3f} - {data_time.average:.3f} \t \
                    Batch time:{batch_time.val:.3f} - {batch_time.average:.3f} \t \
                    Loss: {loss.val:.4f} - {loss.average:.4f}')

        del pred_locs, pred_scores, images, boxes, labels #删除变量，节省内存


def main():
    dataset = PasCalDataset(data_folder)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn,
                             num_workers=workers, pin_memory=True, drop_last=True)
    print('initialize the model!')
    model = tiny_detector(n_classes)
    print(f'model parameters: {sum(p.numel() for p in model.parameters()) / 1000000.0}M')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(model.prior_cxcy)

    model = model.to(device) # 应该先把model放到gpu上，再load_checkpoint
    criterion = criterion.to(device) #移到gpu上！

    if resume:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(666)

    for i in range(start_epoch, epoch):
        train(i, train_loader, model, optimizer, criterion)
        if (i+1) % step_size == 0:
            adjust_learning_rate(optimizer, scale)


        save_checkpoint(checkpoint_path, i, model, optimizer) #TODO 比较loss？ epoch写对！！

def test():
    pass

if __name__ == '__main__':
    main()
