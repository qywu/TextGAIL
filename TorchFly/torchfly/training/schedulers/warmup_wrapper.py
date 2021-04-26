import warnings
from torch.optim.lr_scheduler import _LRScheduler


class WarmupWarpper(_LRScheduler):
    """
    """
    def __init__(self, scheduler, warmup_steps, total_num_update_steps, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.total_num_update_steps = total_num_update_steps
        self.finished = False
        super().__init__(scheduler.optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.total_num_update_steps:
            if not self.finished:
                self.finished = True
            return self.scheduler.get_last_lr()
        elif self.last_epoch < self.warmup_steps:
            multiplier = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [lr * multiplier for lr in self.base_lrs]
        else:
            return self.scheduler.get_last_lr()

    def step(self):
        if self.last_epoch < self.warmup_steps:
            super().step()
        elif self.last_epoch < self.total_num_update_steps:
            self.scheduler.step()
            self._last_lr = self.scheduler.get_last_lr()
            self.last_epoch += 1
        else:
            if self.last_epoch == self.total_num_update_steps:
                warnings.warn("Learning rate scheduler steps have exceeded total_num_update_steps!")
            super().step()

