import torch 
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


def mycopy(target: OrderedDict[str, torch.Tensor], 
           source: OrderedDict[str, torch.Tensor]) -> None:
    for name in target:
        target[name] = source[name].clone().detach()
        target[name].requires_grad_(source[name].requires_grad)

def subtract_(target: OrderedDict[str, torch.Tensor],
              minuend: OrderedDict[str, torch.Tensor], 
              subtrahend: OrderedDict[str, torch.Tensor]) -> None:
    for name in target:
        target[name] = minuend[name].clone().detach() - subtrahend[name].clone().detach()
        target[name].requires_grad_(minuend[name].requires_grad)

def flatten(source: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.flatten() for value in source.values()])

class Client_FRL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, model_wd,
                 train_dl_local = None, test_dl_local = None) -> None:
        
        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.model_wd = model_wd
        self.device = device
        self.net.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.w = OrderedDict({key: value for key, value in self.net.named_parameters()})
        self.w_old = OrderedDict(
            {key : torch.zeros_like(value) for key, value in self.net.named_parameters()}
        )
        self.dw = OrderedDict(
            {key: torch.zeros_like(value) for key, value in self.net.named_parameters()}
        )

    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()

        self.w = OrderedDict(
            {key: value for key, value in self.net.named_parameters()}
        )
        mycopy(target=self.w_old, source=self.w)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.model_wd)
        
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        subtract_(target=self.dw, minuend=self.w, subtrahend=self.w_old)
        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict, strict=False)

    def get_net(self):
        return self.net

    def get_best_acc(self):
        return self.acc_best
    
    def get_count(self):
        return self.count
    
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy