# Train
from efficientnet import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
import os
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

def save_checkpoint_state(epoch, model, optimizer, running_loss):
        checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }
            
        torch.save(checkpoint, "./ckpts/checkpoint.pth.tar")


def load_checkpoint_state(path, device, model, optimizer):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, epoch, optimizer

def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
#        print(images.shape)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        print('Epoch ', epoch, loss.detach().numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #writer.add_scalar('Loss/train', loss, epoch)
    #writer.close()


def main(args):
    args.classes_num = 2
    if args.pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.classes_num, advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, override_params={'num_classes': args.classes_num})

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_dataset = build_data_set(args.image_size, args.train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Resume {}".format(args.resume))
    if args.resume:
        print("Resume last training.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, start_epoch, optimizer = load_checkpoint_state("./ckpts/checkpoint.pth.tar", device, model, optimizer)

    for epoch in range(args.epochs):
        loss = train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar.epoch_%s' % epoch))
            save_checkpoint_state(epoch, model, optimizer, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u"[+]----------------- 图像分类Sample -----------------[+]")
    parser.add_argument('--train-data', default='./data/train', dest='train_data', help='location of train data')
    parser.add_argument('--image-size', default=224, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr',  default=0.001,type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='efficientnet-b0', help='arch type of EfficientNet')
    parser.add_argument('--pretrained', default=True, help='learning rate')
    parser.add_argument('--advprop', default=False, help='advprop')
    parser.add_argument('--resume', action="store_true", help='resume')
    args = parser.parse_args()
    main(args)
