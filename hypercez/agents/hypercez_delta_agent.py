import copy
from collections import Counter

import torch
from torch import nn

from hypercez import make_scheduler, adjust_lr, has_nan, clone_ez_state
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import DecisionModel
from hypercez.agents.hypercez_agent import HyperCEZAgent
from hypercez.hypernet import build_hnet
from hypercez.hypernet.hypercl.utils import ewc_regularizer as ewc
from hypercez.hypernet.hypercl.utils import hnet_regularizer
from hypercez.hypernet.hypercl.utils import optim_step as opstep


class HyperCEZDeltaAgent(HyperCEZAgent):
    def __init__(self, hparams, ez_agent: EZAgent, *hnet_component_names):
        super().__init__(hparams, ez_agent, *hnet_component_names)
        self.__frozen_ez_state = None
        self.alphas = None
        self.alpha_max = 0.2
        self.layerNorms = None

    def init_model(self):
        if len(self.hnet_component_names) == 1 and self.hnet_component_names[0] == "singular":
            self.hnet_map = {"singular": build_hnet(hparams=self.hparams, model=self.ez_agent.model)}
        else:
            for component_name in self.hnet_component_names:
                self.hnet_map[component_name] = build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.model.__getattr__(component_name)
                )
        self.ez_agent.mem_id = 0
        self.__frozen_ez_state = clone_ez_state(self.ez_agent)
        self.alphas = {component_name: nn.Parameter(torch.tensor(1e-3)) for component_name in self.hnet_component_names}
        self.add_task(0)

    def to(self, device=torch.device("cpu")):
        super().to(device)
        self.ez_agent.to(device)
        for hnet_name in self.hnet_component_names:
            self.hnet_map[hnet_name].to(device)
            self.alphas[hnet_name].to(device)
            self.__frozen_ez_state[hnet_name] = {
                k: v.to(device)
                for k, v in self.__frozen_ez_state[hnet_name].items()
            }

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
            d_weights =  hnet_map["singular"](task_id)
            i = 0
            for param_name, _ in ez_agent_model.named_parameters():
                full_state[param_name] = d_weights[i] # Fixme: fix this to work with W = alpha * d_W
                i += 1
            return full_state

        for component_name in self.ez_agent.get_model_list():
            if component_name not in self.hnet_component_names:
                full_state[component_name] = dict(ez_agent_model.__getattr__(component_name).named_parameters())
            else:
                state_dict = {}
                d_weights = hnet_map[component_name](task_id)
                i = 0
                for param_name, orig_param in ez_agent_model.__getattr__(component_name).named_parameters():
                    if param_name in self.layerNorms:
                        state_dict[param_name] = orig_param
                    else:
                        effective_alpha = self.alpha_max * torch.tanh(self.alphas[component_name])
                        state_dict[param_name] = (self.__frozen_ez_state[component_name][param_name] +
                                                  effective_alpha * d_weights[i])
                        assert d_weights[i].requires_grad
                        assert not torch.isnan(d_weights[i]).any(), "NaN in generated weights for " + component_name
                    i += 1
                full_state[component_name] = state_dict

        return full_state

    def add_task(self, task_id, use_prior=False):
        """This is equivalent to augment_model in HyperCRL"""
        assert task_id not in self.seen_tasks, f"Task {task_id} already seen"
        self.seen_tasks.add(task_id)
        targets = {}
        fisher_est_map = {}
        theta_optimizers = {}
        emb_optimizers = {}
        alpha_optimizer = torch.optim.Adam(self.alphas.values(), lr=self.hparams.lr_hyper)
        reg_param_map = {}
        schedulers = {alpha_optimizer: make_scheduler(alpha_optimizer, self.hparams)}
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
            targets, theta_optimizers, emb_optimizers, alpha_optimizer, reg_param_map, fisher_est_map, schedulers, scaler
        )
        return targets, theta_optimizers, emb_optimizers, alpha_optimizer, reg_param_map, fisher_est_map, schedulers, scaler

    def learn(self, task_id: int, verbose=False):
        assert self.training_mode, "Agent not in training mode"
        state_ = self.make_ez_state(task_id, DecisionModel.REANALYZE)
        try:
            batch = self._HyperCEZAgent__memory_manager.make_batch(
                task_id,
                reanalyze_state=state_
            )
        except Exception as e:
            nan_found, componenent = has_nan(state_)
            print(f"Error in state: {nan_found} in {componenent}")
            raise e

        (targets,
         theta_optimizers,
         emb_optimizers,
         alpha_optimizer,
         reg_param_map,
         fisher_est_map,
         schedulers,
         scaler) = self.cl_training_misc[task_id]

        # Adjust Learning Rates
        adjust_lr(
            self.hparams,
            alpha_optimizer,
            self._HyperCEZAgent__memory_manager.get_trained_steps(task_id),
            schedulers[alpha_optimizer]
        )
        for component_name in self.hnet_component_names:
            adjust_lr(
                self.hparams,
                theta_optimizers[component_name],
                self._HyperCEZAgent__memory_manager.get_trained_steps(task_id),
                schedulers[theta_optimizers[component_name]]
            )
            adjust_lr(
                self.hparams,
                emb_optimizers[component_name],
                self._HyperCEZAgent__memory_manager.get_trained_steps(task_id),
                schedulers[emb_optimizers[component_name]]
            )
        if Counter(self.ez_agent.get_model_list()) != Counter(self.hnet_component_names):
            self.ez_agent.adjust_lr(
                self.ez_agent.optimizer,
                self._HyperCEZAgent__memory_manager.get_trained_steps(task_id),
                self.ez_agent.scheduler
            )

        if self._HyperCEZAgent__memory_manager.get_trained_steps() % 30 == 0:
            self.hnet_map_latest = copy.deepcopy(self.hnet_map)
            self.ez_agent.latest_model = copy.deepcopy(self.ez_agent.model)

        if self._HyperCEZAgent__memory_manager.get_trained_steps() % self.hparams.train["self_play_update_interval"] == 0:
            self.hnet_map_self_play = copy.deepcopy(self.hnet_map)
            self.ez_agent.self_play_model = copy.deepcopy(self.ez_agent.model)

        if self._HyperCEZAgent__memory_manager.get_trained_steps() % self.hparams.train["reanalyze_update_interval"] == 0:
            self.hnet_map_reanalyze = copy.deepcopy(self.hnet_map)
            self.ez_agent.reanalyze_model = copy.deepcopy(self.ez_agent.model)


        weighted_loss, log_data = self.ez_agent.calc_loss(
            self.ez_agent.model,
            batch,
            self._HyperCEZAgent__memory_manager.get_item("replay_buffer_map", task_id),
            self._HyperCEZAgent__memory_manager.get_trained_steps(task_id),
            model_state=self.make_ez_state(task_id, DecisionModel.CURRENT)
        )

        self.update_weights(weighted_loss, task_id)
        self._HyperCEZAgent__memory_manager.increment_trained_steps(task_id)

    def update_weights(self, weighted_loss, task_id: int):
        (targets,
         theta_optimizers,
         emb_optimizers,
         alpha_optimizer,
         reg_param_map,
         fisher_est_map,
         schedulers,
         scaler) = self.cl_training_misc[task_id]

        with torch.amp.autocast(self.device.type):
            weighted_loss.register_hook(lambda grad: grad * (1. / self.hparams.train["unroll_steps"]))

        # Zero gradients
        self.ez_agent.optimizer.zero_grad()
        alpha_optimizer.zero_grad()
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
        scaler.unscale_(alpha_optimizer)
        torch.nn.utils.clip_grad_norm_(self.alphas.values(), self.hparams.grad_max_norm)
        scaler.step(alpha_optimizer)
        for component_name in self.hnet_component_names:
            scaler.unscale_(theta_optimizers[component_name])
            torch.nn.utils.clip_grad_norm_(reg_param_map[component_name], self.hparams.grad_max_norm)
            scaler.step(theta_optimizers[component_name])

        scaler.update()

        if self.hparams.model["noisy_net"]:
            self.ez_agent.model.value_policy_model.reset_noise()

        return [scaler]
