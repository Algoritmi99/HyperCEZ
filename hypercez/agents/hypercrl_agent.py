import torch

from hypercez.agents.agent_base import Agent, ActType
from hypercez.util.datautil import DataCollector
from hypercez.hypernet.hypercrl.init_tools import build_model_hnet as build_model
from hypercez.hypernet.hypercrl.control.agent import MPC
from hypercez.agents.random_agent import RandomAgent
from hypercez.hypernet.hypercl.utils import hnet_regularizer as hreg
from hypercez.hypernet.task_loss import TaskLoss, TaskLossMT, TaskLossReplay
from hypercez.hypernet.hypercl.utils import ewc_regularizer as ewc
from hypercez.hypernet.hypercl.utils import si_regularizer as si
from hypercez.hypernet.hypercl.utils import optim_step as opstep



class HyperCRLAgent(Agent):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.collector = DataCollector(hparams)
        self.mnet, self.hnet = build_model(hparams)
        self.agent = MPC(hparams, self.mnet, collector=self.collector, hnet=self.hnet)
        self.rand_agent = RandomAgent(hparams)
        self.seen_tasks = set()
        self.learned_steps = {}
        self.trainer_miscs = {}
        self.total_training_steps = self.hparams.max_iteration // self.hparams.dynamics_update_every

    def to(self, device=torch.device("cpu")):
        super().to(device)
        self.mnet.to(device)
        self.hnet.to(device)

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        if task_id not in self.seen_tasks:
            self.add_task(task_id)
            self.learned_steps[task_id] = 0
        return None, None, self.agent.act(obs, task_id=task_id).detach().cpu().numpy()

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return None, None, self.rand_agent.act(obs, task_id=task_id)

    def is_ready_for_training(self, task_id=None, pbar=None):
        if pbar is not None:
            pbar.total = self.hparams.init_rand_steps
            pbar.n = len(self.collector.states[task_id]) if task_id in self.collector.states else 0
            pbar.refresh()
        return (len(self.collector.states[task_id]) if task_id in self.collector.states else 0) >= self.hparams.init_rand_steps


    def done_training(self, task_id=None, pbar=None):
        if task_id not in self.learned_steps:
            self.learned_steps[task_id] = 0

        if pbar is not None:
            pbar.total = self.total_training_steps
            pbar.n = self.learned_steps[task_id]
            pbar.refresh()
        return self.learned_steps[task_id] >= self.total_training_steps

    def init_model(self):
        self.add_task(0)

    def add_task(self, task_id):
        assert task_id not in self.seen_tasks, "Task already seen"
        # Regularizer targets.
        targets = hreg.get_current_targets(task_id, self.hnet)

        # Add new hypernet embeddings and Loss Function
        self.hnet.add_task(task_id, self.hparams.std_normal_temb)

        if self.hparams.model == "hnet_mt":
            # Loss Function
            mll = TaskLossMT(self.hparams, self.mnet, self.hnet, self.collector, task_id)
        elif self.hparams.model == "hnet_replay":
            mll = TaskLossReplay(self.hparams, self.mnet, self.hnet, self.collector, task_id)
        else:
            mll = TaskLoss(self.hparams, self.mnet)

        # (Re)Put model to GPU
        self.mnet.to(self.device)
        self.hnet.to(self.device)

        # Optimize over the GP model params and likelihood param
        self.mnet.train()
        self.hnet.train()

        # Collect Fisher estimates for the reg computation.
        fisher_ests = None
        if self.hparams.ewc_weight_importance and task_id > 0:
            fisher_ests = []
            n_W = len(self.hnet.target_shapes)
            for t in range(task_id):
                ff = []
                for i in range(n_W):
                    _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                    ff.append(getattr(self.mnet, buff_f_name))
                fisher_ests.append(ff)

        # Register SI buffers for new task
        si_omega = None
        if self.hparams.model == "hnet_si":
            si.si_register_buffer(self.mnet, self.hnet, task_id)
            if task_id > 0:
                si_omega = si.get_si_omega(self.mnet, task_id)

        regularized_params = list(self.hnet.theta)
        if task_id > 0 and self.hparams.plastic_prev_tembs:
            for i in range(task_id):  # for all previous task embeddings
                regularized_params.append(self.hnet.get_task_emb(i))
        theta_optimizer = torch.optim.Adam(regularized_params, lr=self.hparams.lr_hyper)
        # We only optimize the task embedding corresponding to the current task,
        # the remaining ones stay constant.
        emb_optimizer = torch.optim.Adam([self.hnet.get_task_emb(task_id)],
                                         lr=self.hparams.lr_hyper)

        trainer_misc = (targets, mll, theta_optimizer, emb_optimizer, regularized_params,
                        fisher_ests, si_omega)

        self.trainer_miscs[task_id] = trainer_misc
        self.seen_tasks.add(task_id)
        return trainer_misc

    def train(self):
        self.hnet.train()
        self.mnet.train()

    def reset_trained_steps(self):
        pass

    def get_trained_steps(self, task_id=None):
        return 0

    def learn(self, task_id: int, verbose=False):
        train_set, _ = self.collector.get_dataset(task_id)
        # Data Loader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.hparams.bs, shuffle=True,
                                                   drop_last=True, num_workers=self.hparams.num_ds_worker)

        regged_outputs = None

        targets, mll, theta_optimizer, emb_optimizer, regularized_params, \
            fisher_ests, si_omega = self.trainer_miscs[task_id]

        # Whether the regularizer will be computed during training?
        calc_reg = task_id > 0 and self.hparams.beta > 0

        it = 0
        while it < self.hparams.train_dynamic_iters:
            self.mnet.train()
            self.hnet.train()
            for i, data in enumerate(train_loader):
                if len(data) == 3:
                    x_t, a_t, x_tt = data
                    x_t, a_t, x_tt = x_t.to(self.device), a_t.to(self.device), x_tt.to(self.device)
                    X = torch.cat((x_t, a_t), dim=-1)
                else:
                    X, x_tt = data
                    X, x_tt = X.to(self.device), x_tt.to(self.device)

                ### Train theta and task embedding.
                theta_optimizer.zero_grad()
                emb_optimizer.zero_grad()

                weights = self.hnet.forward(task_id)
                if self.hparams.model == "hnet_si":
                    si.si_update_optim_step(self.mnet, weights, task_id)
                    for weight in weights:
                        weight.retain_grad()  # save grad for calculate si path integral

                Y = self.mnet.forward(X, weights)
                # Task-specific loss.
                loss_task = mll(Y, x_tt, weights)
                # We already compute the gradients, to then be able to compute delta
                # theta.
                loss_task.backward(retain_graph=calc_reg,
                                   create_graph=self.hparams.backprop_dt and calc_reg)
                torch.nn.utils.clip_grad_norm_(self.hnet.get_task_emb(task_id), self.hparams.grad_max_norm)

                # The task embedding is only trained on the task-specific loss.
                # Note, the gradients accumulated so far are from "loss_task".
                emb_optimizer.step()

                # SI
                if self.hparams.model == "hnet_si":
                    torch.nn.utils.clip_grad_norm_(weights, self.hparams.grad_max_norm)
                    si.si_update_grad(self.mnet, weights, task_id)

                # Update Regularization
                loss_reg = torch.tensor(0., requires_grad=False)
                dTheta = None
                grad_tloss = None
                if calc_reg:
                    if i % 1000 == 0:  # Just for debugging: displaying grad magnitude.
                        grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                                self.hnet.theta])
                    if self.hparams.no_look_ahead:
                        dTheta = None
                    else:
                        dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                         self.hparams.use_sgd_change, lr=self.hparams.lr_hyper,
                                                         detach_dt=not self.hparams.backprop_dt)

                    if self.hparams.plastic_prev_tembs:
                        dTembs = dTheta[-task_id:]
                        dTheta = dTheta[:-task_id] if dTheta is not None else None
                    else:
                        dTembs = None

                    loss_reg = hreg.calc_fix_target_reg(self.hnet, task_id,
                                                        targets=targets, dTheta=dTheta, dTembs=dTembs, mnet=self.mnet,
                                                        inds_of_out_heads=regged_outputs,
                                                        fisher_estimates=fisher_ests,
                                                        si_omega=si_omega)

                    loss_reg = loss_reg * self.hparams.beta * Y.size(0)

                    loss_reg.backward()

                    if grad_tloss is not None:  # Debug
                        grad_full = torch.cat([d.grad.view(-1) for d in self.hnet.theta])
                        # Grad of regularizer.
                        grad_diff = grad_full - grad_tloss
                        grad_diff_norm = torch.norm(grad_diff, 2)

                        # Cosine between regularizer gradient and task-specific
                        # gradient.
                        if dTheta is None:
                            dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                             self.hparams.use_sgd_change, lr=self.hparams.lr_hyper,
                                                             detach_dt=not self.hparams.backprop_dt)
                        dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                        grad_cos = torch.nn.functional.cosine_similarity(grad_diff.view(1, -1),
                                                                         dT_vec.view(1, -1))

                        grad_tloss = (grad_tloss, grad_full, grad_diff_norm, grad_cos)

                torch.nn.utils.clip_grad_norm_(regularized_params, self.hparams.grad_max_norm)
                theta_optimizer.step()

                it += 1
                if it >= self.hparams.train_dynamic_iters:
                    break
        self.learned_steps[task_id] += 1

    def eval(self):
        self.mnet.eval()
        self.hnet.eval()

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        self.collector.add(x_t, u_t, reward, x_tt, task_id)

    def reset(self, obs):
        self.agent.reset()
