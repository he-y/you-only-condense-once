import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from misc.utils import random_indices, rand_bbox

class Trainer(object):
    """
    Helper class for training.
    """

    def __init__(self, aim_run=None, mixup=False, nclass=10):
        self.aim_run = aim_run
        self.mixup = mixup
        self.nclass = nclass

    """
    Dataset need to be an index dataset.
    Set remaining_iterations to -1 to ignore this argument.
    """
    def train(self, epoch, remaining_iterations, model, dataloader, optimizer, criterion, scheduler, device, aug=None, TD_logger=None, log_interval=None, printlog=False):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()
        if printlog:
            print('*' * 26)
        for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
            inputs = inputs.to(torch.float32)
            targets = targets.to(torch.long)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            if aug:
                with torch.no_grad():
                    inputs = aug(inputs)
            
            r = np.random.rand(1)
            if r < 0.5 and self.mixup:
                print('doing mixup')
                # generate mixed sample
                lam = np.random.beta(1.0, 1.0)
                rand_index = random_indices(targets, nclass=self.nclass)

                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                outputs = model(inputs)
                loss = criterion(outputs, targets) * ratio + criterion(outputs, target_b) * (1. - ratio)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
                if self.aim_run:
                    self.aim_run.track(scheduler.get_lr()[0], name='lr', epoch=epoch)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

            if TD_logger:
                log_tuple = {
                'epoch': epoch,
                'iteration': batch_idx,
                'idx': idx.type(torch.long).clone(),
                'output': F.log_softmax(outputs, dim=1).detach().cpu().type(torch.half)
                }
                TD_logger.log_tuple(log_tuple)

            if printlog and log_interval and batch_idx % log_interval == 0:
                print(f"{batch_idx}/{len(dataloader)}")
                print(f'>> batch_idx [{batch_idx}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')

            remaining_iterations -= 1
            if remaining_iterations == 0:
                if printlog: print("Exit early in epoch training.")
                break

        if self.aim_run:
            self.aim_run.track(train_loss, name='train_loss', epoch=epoch)
            self.aim_run.track(correct/total * 100, name='train_acc', epoch=epoch)
        
        if printlog:
            print(f'>> Epoch [{epoch}]: Loss: {train_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'>> Epoch [{epoch}]: Training Accuracy: {correct/total * 100:.2f}')
            print(f'>> Epoch [{epoch}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')

    def test(self, model, dataloader, criterion, device, aug=None, epoch=0, log_interval=None,  printlog=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()

        if printlog: print('======= Testing... =======')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                # if aug:
                #     with torch.no_grad():
                #         inputs = aug(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.shape[0]
                correct += predicted.eq(targets).sum().item()

                # if printlog and log_interval and batch_idx % log_interval == 0:
                #     print(batch_idx)

        if self.aim_run:
            self.aim_run.track(test_loss, name='test_loss', epoch=epoch)
            self.aim_run.track(correct/total * 100, name='test_acc', epoch=epoch)

        if printlog:
            print(f'Loss: {test_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'Test Accuracy: {correct/total * 100:.2f}')

        print(f'>> Test time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
        return test_loss, correct / total