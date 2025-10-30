import copy
from enum import IntEnum

import torch
import torch.nn as nn

from hypercez.agents import EZAgent
from hypercez.agents.agent_base import Agent, ActType
from hypercez.agents.ez_agent import AgentType
from hypercez.hypernet import build_hnet
from hypercez.util.hypercez_data_mng import HyperCEZDataManager
from hypercez.hypernet.hypercl.utils import hnet_regularizer as hreg
from hypercez.hypernet.hypercl.utils import ewc_regularizer as ewc
from hypercez.hypernet.hypercl.utils import si_regularizer as si
from hypercez.hypernet.hypercl.utils import optim_step as opstep


class AgentCtrlType(IntEnum):
    DISCRETE = 0
    IMG_BASED = 1
    CONTINUOUS = 2

class HNetType(IntEnum):
    HNET = 0
    HNET_SI = 1
    HNET_EWC = 2
    HNET_MT = 3
    HNET_REPLAY = 4

class HyperCEZAgent(Agent):
    def __init__(self,
                 hparams,
                 ez_agent: EZAgent = None,
                 hnet_type: HNetType = HNetType.HNET,
                 hnet_map: dict[str, nn.Module] = None,
                 ctrl_type=AgentCtrlType.CONTINUOUS,
                 *args: str,
                 ):
        """
        An EfficientZeroV2 agent that uses hypernetworks for continual learning on each of its components.

        The keys of hnet_map or the strings provided as *args can include one or more of the following:
        [
            "representation_model",
            "dynamics_model",
            "reward_prediction_model",
            "value_policy_model",
            "projection_model",
            "projection_head_model"
        ]
        :param hparams: HParams object with added values
        :param ez_agent: EZAgent object correctly initialized
        :param hnet_map: a map with keys being one of the names mentioned above or "all" and values being nn.Module objects
        :param ctrl_type: Specifies the type of controller to be used (img based, discrete or continuous)
        :param args: a listing of names of agent including one or more of the names listed above or "all", to create hnets for if hnet_map is not provided.
        """
        super().__init__(hparams)
        self.reg_targets = None
        self.theta_optims = None
        self.emb_optims = None
        self.regularized_params = None
        self.fisher_ests = None
        self.si_omegas = None
        self.ez_agent = ez_agent
        self.hnet_type = hnet_type
        self.hnet_map = hnet_map
        self.control_type = ctrl_type
        self.hnet_component_names = list(args) if hnet_map is None else list(hnet_map.keys())
        self.__memory_manager = HyperCEZDataManager(self.ez_agent)
        self.__training_mode = False
        self.__dtype = None
        self.learn_called = 0

    def init_model(self):
        if self.ez_agent is None:
            self.ez_agent = EZAgent(
                self.hparams,
                AgentType(self.control_type.value)
            )
        if self.hnet_map is None:
            assert len(self.hnet_component_names) > 0, "either hnet_map or hnet_items should be provided!"
            self.hnet_map = {}
            for hnet_comp in self.hnet_component_names:
                hnet = build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.get_model(hnet_comp)
                )
                for task in range(self.hparams.num_tasks):
                    hnet.add_task(task, self.hparams.std_normal_temb)
                self.hnet_map[hnet_comp] = hnet


        assert self.ez_agent is not None and self.hnet_map is not None

    def to(self, device=torch.device("cpu")):
        super().to(device)
        self.ez_agent.to(device)
        for hnet_name in self.hnet_component_names:
            self.hnet_map[hnet_name].to(device)

    def is_ready_for_training(self, task_id=None, pbar=None):
        return self.__memory_manager.is_ready_for_training(task_id=task_id, pbar=pbar)

    def done_training(self, task_id=None, pbar=None):
        self.__memory_manager.done_training(task_id=task_id, pbar=pbar)

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        assert self.hnet_map is not None
        for hnet_name in self.hnet_component_names:
            new_weights = self.hnet_map[hnet_name](task_id)
            with torch.no_grad():
                for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                    assert p.shape == w.shape
                    w = w.to(device=p.device, dtype=p.dtype, non_blocking=True)
                    w.requires_grad_(p.requires_grad)
                    p.copy_(w)
                    if self.__dtype is None:
                        self.__dtype = torch.float32
            for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                w = w.to(device=p.device, dtype=p.dtype)
                if not torch.equal(p, w):
                    print("obs:", obs)
                    print("task_id:", task_id)
                    print("p:", p)
                    print("w:", w)
                    assert False
                # assert torch.equal(p, w)
        return self.ez_agent.act(obs, task_id, act_type)

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return self.act(obs, task_id, act_type)

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        self.__memory_manager.collect(x_t, u_t, reward, x_tt, task_id, done=done)

    def eval(self):
        [self.hnet_map[hnet_name].eval() for hnet_name in self.hnet_component_names]
        self.ez_agent.eval()

    def train(self):
        # structure of keys: hnet_name -> task_id -> object
        self.reg_targets = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.theta_optims = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.emb_optims = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.regularized_params = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.fisher_ests = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.si_omegas = {hnet_name: {} for hnet_name in self.hnet_component_names}

        # put hnets and ez_agent to training mode
        [self.hnet_map[hnet_name].train() for hnet_name in self.hnet_component_names]
        self.ez_agent.train()

        for hnet_name in self.hnet_component_names:
            for task_id in range(self.hparams.num_tasks):
                self.reg_targets[hnet_name][task_id] = hreg.get_current_targets(task_id, self.hnet_map[hnet_name])

                self.fisher_ests[hnet_name][task_id] = None
                if self.hparams.ewc_weight_importance and task_id > 0:
                    self.fisher_ests[hnet_name][task_id] = []
                    n_W = len(self.hnet_map[hnet_name].target_shapes)
                    for t in range(task_id):
                        ff = []
                        for i in range(n_W):
                            _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                            ff.append(getattr(self.ez_agent.model.__getattr__(hnet_name), buff_f_name))
                        self.fisher_ests[hnet_name][task_id].append(ff)

                self.si_omegas[hnet_name][task_id] = None
                if self.hnet_type == HNetType.HNET_SI:
                    si.si_register_buffer(
                        self.ez_agent.model.__getattr__(hnet_name),
                        self.hnet_map[hnet_name],
                        task_id
                    )
                    if task_id > 0:
                        self.si_omegas[hnet_name][task_id] = si.get_si_omega(
                            self.ez_agent.model.__getattr__(hnet_name),
                            task_id
                        )

                self.regularized_params[hnet_name][task_id] = list(self.hnet_map[hnet_name].theta)
                if task_id > 0 and self.hparams.plastic_prev_tembs:
                    for i in range(task_id):  # for all previous task embeddings
                        self.regularized_params[hnet_name][task_id].append(self.hnet_map[hnet_name].get_task_emb(i))

                self.theta_optims[hnet_name][task_id] = torch.optim.Adam(
                    self.regularized_params[hnet_name][task_id],
                    lr=self.hparams.lr_hyper
                )

                self.emb_optims[hnet_name][task_id] = torch.optim.Adam(
                    [self.hnet_map[hnet_name].get_task_emb(task_id)],
                    lr=self.hparams.lr_hyper
                )

        self.learn_called = 0
        self.__training_mode = True

    def learn(self, task_id):
        assert self.__training_mode
        batch = self.__memory_manager.make_batch(task_id)

        # Whether the regularizer will be computed during training?
        calc_reg = task_id > 0 and self.hparams.beta > 0

        # ready optimizers
        for hnet_name in self.hnet_component_names:
            self.theta_optims[hnet_name][task_id].zero_grad()
            self.emb_optims[hnet_name][task_id].zero_grad()

        # forward pass through hypernets and create model state
        full_state = {}
        for hnet_name in self.hnet_component_names:
            new_weights = self.hnet_map[hnet_name](task_id)

            if self.hnet_type == HNetType.HNET_SI:
                # save grad for calculate si path integral if so
                si.si_update_optim_step(self.ez_agent.model.__getattr__(hnet_name), new_weights, task_id)
                for weight in new_weights:
                    weight.retain_grad()

            model_state = {}
            for i, param_name in enumerate(
                    [param_name for param_name, _ in self.ez_agent.model.__getattr__(hnet_name).named_parameters()]
            ):
                model_state[param_name] = new_weights[i].requires_grad_() if not new_weights[i].requires_grad else new_weights[i]
            full_state[hnet_name] = model_state

        # calculate loss
        loss_task, _ = self.ez_agent.calc_loss(
            self.ez_agent.model,
            batch,
            copy.deepcopy(self.__memory_manager.get_item('replay_buffer_map', task_id)),
            self.ez_agent.trained_steps,
            model_state=full_state
        )
        with torch.amp.autocast(self.device.type):
            loss_task.register_hook(lambda grad: grad * (1. / self.hparams.train["unroll_steps"]))

        self.ez_agent.scaler.scale(loss_task).backward(
            retain_graph=calc_reg,
            create_graph=self.hparams.backprop_dt and calc_reg,
        )

        # loss_task.backward(retain_graph=calc_reg,
        #                    create_graph=self.hparams.backprop_dt and calc_reg)

        for hnet_name in self.hnet_component_names:
            weights = [v for _, v in full_state[hnet_name].items()]
            # clip grads
            torch.nn.utils.clip_grad_norm_(self.hnet_map[hnet_name].get_task_emb(task_id), self.hparams.grad_max_norm)

            # Note, the gradients accumulated so far are from "loss_task".
            self.emb_optims[hnet_name][task_id].step()

            # SI
            if self.hnet_type == HNetType.HNET_SI:
                torch.nn.utils.clip_grad_norm_(
                    weights,
                    self.hparams.grad_max_norm
                )
                si.si_update_grad(self.ez_agent.model.__getattr__(hnet_name), weights, task_id)

            # Update Regularization
            grad_tloss = None

            if calc_reg:
                if self.learn_called % 1000 == 0:  # Just for debugging: displaying grad magnitude.
                    grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                            self.hnet_map[hnet_name].theta])
                if self.hparams.no_look_ahead:
                    dTheta = None
                else:
                    dTheta = opstep.calc_delta_theta(self.theta_optims[hnet_name][task_id],
                                                     self.hparams.use_sgd_change, lr=self.hparams.lr_hyper,
                                                     detach_dt=not self.hparams.backprop_dt)

                if self.hparams.plastic_prev_tembs:
                    dTembs = dTheta[-task_id:]
                    dTheta = dTheta[:-task_id] if dTheta is not None else None
                else:
                    dTembs = None

                loss_reg = hreg.calc_fix_target_reg(self.hnet_map[hnet_name],
                                                    task_id,
                                                    targets=self.reg_targets[hnet_name][task_id],
                                                    dTheta=dTheta,
                                                    dTembs=dTembs,
                                                    mnet=self.ez_agent.model.__getattr__(hnet_name),
                                                    inds_of_out_heads=None,
                                                    fisher_estimates=self.fisher_ests[hnet_name][task_id],
                                                    si_omega=self.si_omegas[hnet_name][task_id])

                loss_reg = loss_reg * self.hparams.beta * self.hparams.train["batch_size"]

                loss_reg.backward()

                if grad_tloss is not None:  # Debug
                    grad_full = torch.cat([d.grad.view(-1) for d in self.hnet_map[hnet_name].theta])
                    # Grad of regularizer.
                    grad_diff = grad_full - grad_tloss
                    grad_diff_norm = torch.norm(grad_diff, 2)

                    # Cosine between regularizer gradient and task-specific
                    # gradient.
                    if dTheta is None:
                        dTheta = opstep.calc_delta_theta(
                            self.theta_optims[hnet_name][task_id],
                            self.hparams.use_sgd_change,
                            lr=self.hparams.lr_hyper,
                            detach_dt=not self.hparams.backprop_dt
                        )
                    dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                    grad_cos = torch.nn.functional.cosine_similarity(grad_diff.view(1, -1),
                                                                     dT_vec.view(1, -1))

            torch.nn.utils.clip_grad_norm_(self.regularized_params[hnet_name][task_id], self.hparams.grad_max_norm)
            self.theta_optims[hnet_name][task_id].step()
            self.learn_called += 1
            self.__memory_manager.increment_trained_steps(task_id)

    def reset(self, obs):
        self.ez_agent.reset(obs)


