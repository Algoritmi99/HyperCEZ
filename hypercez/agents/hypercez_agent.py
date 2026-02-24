import copy
from collections import Counter

import torch

from hypercez import check_finite, make_scheduler, adjust_lr
from hypercez.agents import EZAgent
from hypercez.agents.agent_base import Agent, ActType
from hypercez.agents.ez_agent import DecisionModel
from hypercez.hypernet import build_hnet
from hypercez.hypernet.hypercl.utils import hnet_regularizer
from hypercez.hypernet.hypercl.utils import ewc_regularizer as ewc
from hypercez.util.hypercez_data_mng import HyperCEZDataManager
from hypercez.hypernet.hypercl.utils import optim_step as opstep


class HyperCEZAgent(Agent):
    def __init__(self, hparams, ez_agent: EZAgent, *hnet_component_names):
        super().__init__(hparams)
        self.hparams = hparams
        self.ez_agent = ez_agent
        self.hnet_map = {}
        self.hnet_map_self_play = None
        self.hnet_map_latest = None
        self.hnet_map_reanalyze = None
        self.hnet_component_names = list(hnet_component_names)
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "all":
            self.hnet_component_names = ez_agent.get_model_list()
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "none":
            self.hnet_component_names = []
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "default":
            self.hnet_component_names = [
                "representation_model",
                "dynamics_model",
                "reward_prediction_model",
                "value_policy_model"
            ]
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "singular":
            self.hnet_component_names = ["singular"]

        self.training_mode = False
        self.schedulers = None
        self.seen_tasks:set[int] = set()
        self.cl_training_misc = {}
        self.__memory_manager = HyperCEZDataManager(self.ez_agent)

    def init_model(self):
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "singular":
            self.hnet_map = {"singular": build_hnet(hparams=self.hparams, model=self.ez_agent.model)}
        else:
            for component_name in self.hnet_component_names:
                self.hnet_map[component_name] = build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.model.__getattr__(component_name)
                )
        self.add_task(0)
        self.match_ez_params(0)
        self.ez_agent.mem_id = 0

    def to(self, device=torch.device("cpu")):
        super().to(device)
        self.ez_agent.to(device)
        for hnet_name in self.hnet_component_names:
            self.hnet_map[hnet_name].to(device)

    def is_ready_for_training(self, task_id=None, pbar=None):
        return self.__memory_manager.is_ready_for_training(task_id=task_id, pbar=pbar)

    def done_training(self, task_id=None, pbar=None):
        return self.__memory_manager.done_training(task_id=task_id, pbar=pbar)

    def reset(self, obs):
        self.ez_agent.reset(obs)

    def match_ez_params(self, task_id, tol=1e-6, max_steps=5000):
        """Match HNET params to produce EZ params supervised"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(torch.device(device))

        print("Matching EZ params by HNet Params on", device)
        for hnet_name in self.hnet_component_names:
            print("Matching", hnet_name)
            with torch.no_grad():
                if hnet_name == "singular":
                    target_weights = [p.detach().clone().to(self.device) for p in self.ez_agent.model.parameters()]
                else:
                    target_weights = [
                        p.detach().clone().to(self.device) for p in self.ez_agent.model.__getattr__(hnet_name).parameters()
                    ]

            if hnet_name == "singular":
                opt = torch.optim.Adam(self.hnet_map.theta, lr=1e-3)
            else:
                opt = torch.optim.Adam(self.hnet_map[hnet_name].theta, lr=1e-3)

            step = 0
            while True:
                opt.zero_grad()
                pred_weights = self.hnet_map[hnet_name](task_id)
                assert len(pred_weights) == len(target_weights), "Mismatch in number of tensors"

                loss = torch.tensor(0.0).to(self.device)

                for pw, tw in zip(pred_weights, target_weights):
                    # Ensure shapes are identical
                    assert pw.shape == tw.shape, f"Shape mismatch: {pw.shape} vs {tw.shape}"
                    loss += torch.nn.functional.mse_loss(pw, tw)

                loss.backward()
                opt.step()

                if step % 100 == 0:
                    print(f"step {step:5d}  loss {loss.item():.6e}")

                if loss.item() < tol:
                    print(f"Stopping, loss below tolerance: {loss.item():.6e}")
                    break

                step += 1
                if step >= max_steps:
                    print(f"Reached max_steps={max_steps}, final loss {loss.item():.6e}")
                    break
            check_finite(self.hnet_map[hnet_name])
            del opt
            del target_weights

    def make_ez_state(self, task_id, decision_model: DecisionModel = DecisionModel.SELF_PLAY):
        if decision_model == DecisionModel.SELF_PLAY:
            hnet_map = self.hnet_map_self_play
            ez_agent_model = self.ez_agent.self_play_model
        elif decision_model == DecisionModel.LATEST:
            hnet_map = self.hnet_map_latest
            ez_agent_model = self.ez_agent.latest_model
        elif decision_model == DecisionModel.REANALYZE:
            hnet_map = self.hnet_map_reanalyze
            ez_agent_model = self.ez_agent.reanalyze_model
        else:
            hnet_map = self.hnet_map
            ez_agent_model = self.ez_agent.model
        if hnet_map is None:
            hnet_map = self.hnet_map

        full_state = {}
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "singular":
            weights =  hnet_map["singular"](task_id)
            i = 0
            for param_name, _ in ez_agent_model.named_parameters():
                full_state[param_name] = weights[i]
                i += 1
            return full_state

        for component_name in self.ez_agent.get_model_list():
            if component_name not in self.hnet_component_names:
                full_state[component_name] = dict(ez_agent_model.__getattr__(component_name).named_parameters())
            else:
                state_dict = {}
                weights = hnet_map[component_name](task_id)
                i = 0
                for param_name, _ in ez_agent_model.__getattr__(component_name).named_parameters():
                    state_dict[param_name] = weights[i]
                    assert not torch.isnan(weights[i]).any(), "NaN in generated weights for " + component_name
                    i += 1
                full_state[component_name] = state_dict
        return full_state

    def act(self, obs, task_id=0, act_type=ActType.INITIAL, decision_model: DecisionModel = DecisionModel.SELF_PLAY):
        if task_id not in self.seen_tasks:
            self.add_task(task_id)
        state = self.make_ez_state(task_id, decision_model=decision_model)
        return self.ez_agent.act(obs, task_id=task_id, act_type=act_type, model_state=state)

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return self.act(obs, task_id, act_type)

    def train(self):
        self.ez_agent.train()
        for hnet_name in self.hnet_component_names:
            self.hnet_map[hnet_name].train()
        self.training_mode = True

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        self.__memory_manager.collect(x_t, u_t, reward, x_tt, task_id, done=done)

    def eval(self):
        [self.hnet_map[hnet_name].eval() for hnet_name in self.hnet_component_names]
        self.ez_agent.eval()

    def add_task(self, task_id, use_prior=False):
        """This is equivalent to augment_model in HyperCRL"""
        assert task_id not in self.seen_tasks, f"Task {task_id} already seen"
        self.seen_tasks.add(task_id)
        targets = {}
        fisher_est_map = {}
        theta_optimizers = {}
        emb_optimizers = {}
        reg_param_map = {}
        schedulers = {}
        scaler = torch.amp.GradScaler()
        for component_name in self.hnet_component_names:
            # Regularizer Targets
            targets[component_name] = hnet_regularizer.get_current_targets(task_id, self.hnet_map[component_name])

            # add new hypernet task embeddings
            self.hnet_map[component_name].add_task(task_id, self.hparams.std_normal_temb, use_prior=use_prior)

            # Collect Fisher estimates for the reg computation.
            fisher_ests = None
            if self.hparams.ewc_weight_importance and task_id > 0:
                fisher_ests = []
                n_W = len(self.hnet_map[component_name].target_shapes)
                for t in range(task_id):
                    ff = []
                    for i in range(n_W):
                        _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                        ff.append(getattr(
                            self.ez_agent.model.__getattr__(component_name) if component_name != "singular" else self.ez_agent.model,
                            buff_f_name,
                            None
                        ))
                    fisher_ests.append(ff)
            fisher_est_map[component_name] = fisher_ests

            regularized_params = list(self.hnet_map[component_name].theta)
            if task_id > 0 and self.hparams.plastic_prev_tembs:
                for i in range(task_id):  # for all previous task embeddings
                    regularized_params.append(self.hnet_map[component_name].get_task_emb(i))
            theta_optimizers[component_name] = torch.optim.Adam(regularized_params, lr=self.hparams.lr_hyper)
            emb_optimizers[component_name] = torch.optim.Adam([self.hnet_map[component_name].get_task_emb(task_id)],
                                             lr=self.hparams.lr_hyper)
            reg_param_map[component_name] = regularized_params
            schedulers[theta_optimizers[component_name]] = make_scheduler(theta_optimizers[component_name], self.hparams)
            schedulers[emb_optimizers[component_name]] = make_scheduler(emb_optimizers[component_name], self.hparams)

        self.cl_training_misc[task_id] = (
            targets, theta_optimizers, emb_optimizers, reg_param_map, fisher_est_map, schedulers, scaler
        )
        return targets, theta_optimizers, emb_optimizers, reg_param_map, fisher_est_map, schedulers, scaler

    def learn(self, task_id: int, verbose=False):
        assert self.training_mode, "Agent not in training mode"
        batch = self.__memory_manager.make_batch(
            task_id,
            reanalyze_state=self.make_ez_state(task_id, DecisionModel.REANALYZE)
        )

        (targets,
         theta_optimizers,
         emb_optimizers,
         reg_param_map,
         fisher_est_map,
         schedulers,
         scaler) = self.cl_training_misc[task_id]

        # Adjust Learning Rates
        for component_name in self.hnet_component_names:
            adjust_lr(
                self.hparams,
                theta_optimizers[component_name],
                self.__memory_manager.get_trained_steps(task_id),
                schedulers[theta_optimizers[component_name]]
            )
            adjust_lr(
                self.hparams,
                emb_optimizers[component_name],
                self.__memory_manager.get_trained_steps(task_id),
                schedulers[emb_optimizers[component_name]]
            )
        if Counter(self.ez_agent.get_model_list()) != Counter(self.hnet_component_names):
            self.ez_agent.adjust_lr(
                self.ez_agent.optimizer,
                self.__memory_manager.get_trained_steps(task_id),
                self.ez_agent.scheduler
            )

        if self.__memory_manager.get_trained_steps() % 30 == 0:
            self.hnet_map_latest = copy.deepcopy(self.hnet_map)
            self.ez_agent.latest_model = copy.deepcopy(self.ez_agent.model)

        if self.__memory_manager.get_trained_steps() % self.hparams.train["self_play_update_interval"] == 0:
            self.hnet_map_self_play = copy.deepcopy(self.hnet_map)
            self.ez_agent.self_play_model = copy.deepcopy(self.ez_agent.model)

        if self.__memory_manager.get_trained_steps() % self.hparams.train["reanalyze_update_interval"] == 0:
            self.hnet_map_reanalyze = copy.deepcopy(self.hnet_map)
            self.ez_agent.reanalyze_model = copy.deepcopy(self.ez_agent.model)


        weighted_loss, log_data = self.ez_agent.calc_loss(
            self.ez_agent.model,
            batch,
            self.__memory_manager.get_item("replay_buffer_map", task_id),
            self.__memory_manager.get_trained_steps(task_id),
        )

        self.update_weights(weighted_loss, task_id)
        self.__memory_manager.increment_trained_steps(task_id)

    def update_weights(self, weighted_loss, task_id: int):
        (targets,
         theta_optimizers,
         emb_optimizers,
         reg_param_map,
         fisher_est_map,
         schedulers,
         scaler) = self.cl_training_misc[task_id]

        with torch.amp.autocast(self.device.type):
            weighted_loss.register_hook(lambda grad: grad * (1. / self.hparams.train["unroll_steps"]))

        # Zero gradients
        self.ez_agent.optimizer.zero_grad()
        for component_name in self.hnet_component_names:
            theta_optimizers[component_name].zero_grad()
            emb_optimizers[component_name].zero_grad()

        # scaled backward
        scaler.scale(weighted_loss).backward(
            retain_graph=task_id > 0 and self.hparams.beta > 0,
            create_graph=self.hparams.backprop_dt and task_id > 0 and self.hparams.beta > 0
        )

        # unscale gradients
        scaler.unscale_(self.ez_agent.optimizer)
        for component_name in self.hnet_component_names:
            # scaler.unscale_(theta_optimizers[component_name])
            scaler.unscale_(emb_optimizers[component_name])

        # clip gradients on embedding params and ezv2 params
        torch.nn.utils.clip_grad_norm_(self.ez_agent.model.parameters(), self.hparams.train["max_grad_norm"])
        for component_name in self.hnet_component_names:
            torch.nn.utils.clip_grad_norm_(
                self.hnet_map[component_name].get_task_emb(task_id), self.hparams.grad_max_norm
            )

        # step embedding optimizer and ezv2 optimizer
        scaler.step(self.ez_agent.optimizer)
        for component_name in self.hnet_component_names:
            scaler.step(emb_optimizers[component_name])


        # Compute Regularization loss
        if task_id > 0 and self.hparams.beta > 0:
            for component_name in self.hnet_component_names:
                if self.hparams.no_look_ahead:
                    d_theta = None
                else:
                    d_theta = opstep.calc_delta_theta(
                        theta_optimizers[component_name],
                        self.hparams.use_sgd_change,
                        lr=self.hparams.lr_hyper,
                        detach_dt=not self.hparams.backprop_dt
                    )

                if self.hparams.plastic_prev_tembs:
                    d_tembs = d_theta[-task_id:]
                    d_theta = d_theta[:-task_id] if d_theta is not None else None
                else:
                    d_tembs = None

                loss_reg = hnet_regularizer.calc_fix_target_reg(
                    self.hnet_map[component_name],
                    task_id,
                    targets=targets[component_name],
                    dTheta=d_theta,
                    dTembs=d_tembs,
                    mnet=self.ez_agent.model if component_name == "singular" else self.ez_agent.model.__getattr__(component_name),
                    fisher_estimates=fisher_est_map[component_name],
                )

                scaler.scale(loss_reg).backward()

        # unscale, clip, step on theta optimizers
        for component_name in self.hnet_component_names:
            scaler.unscale_(theta_optimizers[component_name])
            torch.nn.utils.clip_grad_norm_(reg_param_map[component_name], self.hparams.grad_max_norm)
            scaler.step(theta_optimizers[component_name])

        scaler.update()

        if self.hparams.model["noisy_net"]:
            self.ez_agent.model.value_policy_model.reset_noise()

        return [scaler]
