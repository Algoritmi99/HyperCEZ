import torch


class TaskLoss(torch.nn.Module):
    def __init__(self, hparams, mnet):
        super(TaskLoss, self).__init__()
        self.out_var = hparams.out_var
        self.y_dim: int = hparams.out_dim
        self.mlp_var_minmax = hparams.mlp_var_minmax

        self.reg_norm = 1
        self.reg_lambda = hparams.reg_lambda
        self.mnet = mnet

    def regularize(self, weights):
        if self.reg_lambda == 0:
            return 0

        loss = 0
        for weight in weights:
            loss = weight.norm(self.reg_norm)

        return loss

    def reg_logvar(self, weights):
        if self.mlp_var_minmax:
            max_logvar = self.mnet.mlp_max_logvar
            min_logvar = self.mnet.mlp_min_logvar
        else:
            max_logvar = weights[0]
            min_logvar = weights[1]

        # Regularize max/min var
        loss = 0.01 * (max_logvar.sum() - min_logvar.sum())
        return loss

    def forward(self, pred, gt, weights, add_reg_logvar=True):
        if self.out_var:
            mu, logvar = torch.split(pred, self.y_dim, dim=-1)

            # Compute loss of a task (i.e during evaluation)
            inv_var = torch.exp(-logvar)
            loss = ((mu - gt) ** 2) * inv_var + logvar
            loss = loss.sum() / self.y_dim

            if add_reg_logvar:
                loss += self.reg_logvar(weights)

        else:
            loss = torch.nn.functional.mse_loss(pred, gt, reduction='sum')
            loss = loss / self.y_dim

        loss += self.regularize(weights)

        return loss


class TaskLossMT(TaskLoss):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet)
        self.old_data_iter = None
        self.old_data = None
        self.hnet = hnet
        self.task_id = task_id
        self.gpuid = hparams.gpuid

        self.add_trainset(collector, task_id, hparams)

    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs,
                                                       shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter

    def replay(self, tid):
        loader_it = self.old_data_iter[tid]
        try:
            data = next(loader_it)
        except StopIteration:
            # Reset the dataloader iterable
            loader_it = iter(self.old_data[tid])
            self.old_data_iter[tid] = loader_it
            data = next(loader_it)

        x_t, a_t, x_tt = data
        x_t, a_t, x_tt = x_t.to(self.gpuid), a_t.to(self.gpuid), x_tt.to(self.gpuid)

        # Forward Pass
        X = torch.cat((x_t, a_t), dim=-1)
        weights = self.hnet.forward(tid)
        Y = self.mnet.forward(X, weights)

        # Task-specific loss.
        loss_task = super().forward(Y, x_tt, weights, add_reg_logvar=False)
        return loss_task

    def forward(self, pred, gt, weights, add_reg_logvar=True):
        loss = super().forward(pred, gt, weights)
        for tid in range(0, self.task_id):
            loss += self.replay(tid)
        return loss


class TaskLossReplay(TaskLossMT):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet, hnet, collector, task_id)

    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        M = hparams.bs // task_id if task_id > 0 else 0
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=M,
                                                       shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter
