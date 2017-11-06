#coding:utf8
import torch
import torch.nn.functional  as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from config import opt

from loader import get_loader
from models import get_model

def test(**kwargs):
    opt.parse(kwargs)
    test_loader = DataLoader(get_loader(opt.dataset_name, opt.data_path, train=False), batch_size=opt.test_batch_size, shuffle=True)
    model = get_model(opt.model_name, img_rows=opt.img_rows, num_classes=opt.num_classes)
    model.cuda(0)
    model.eval()
    model.load(opt.model_save_path)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(0), target.cuda(0)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print ('Test set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))


def train(**kwargs):
    opt.parse(kwargs)
    train_loader = DataLoader(get_loader(opt.dataset_name, opt.data_path, train=True), batch_size=opt.batch_size, shuffle=True)
    model = get_model(opt.model_name, img_rows=opt.img_rows, num_classes=opt.num_classes)
    model.cuda(0)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    for epoch in range(1, opt.max_epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(0), target.cuda(0)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0]))
        if epoch % opt.save_freq == 0:
            model.save()

if __name__ == '__main__':
    import fire
    fire.Fire()
