from typing import Dict
import numpy as np
import qpsolvers
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward, avalanche_forward_base
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class GEMPluginRAADL(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.memory_strength = memory_strength

        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_fc: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()
        self.memory_tid: Dict[int, Tensor] = dict()

        self.G: Tensor = torch.empty(0)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if strategy.clock.train_exp_counter > 0:
            G = []
            strategy.model.train()
            for t in range(strategy.clock.train_exp_counter):
                strategy.model.train()
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                fcref = self.memory_fc[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, fcref, self.memory_tid[t])
                loss = strategy._criterion(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten()
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=strategy.device)
                            )
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        tot = 0
        for mbatch in dataloader:
            x, fc, y, tid = mbatch[0], mbatch[1], mbatch[2], mbatch[-1]
            if tot + x.size(0) <= self.patterns_per_experience:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_fc[t] = fc.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_fc[t] = torch.cat((self.memory_fc[t], fc), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_fc[t] = fc[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    self.memory_fc[t] = torch.cat((self.memory_fc[t], fc[:diff]), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat(
                        (self.memory_tid[t], tid[:diff]), dim=0
                    )
                break
            tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        # solution with old quadprog library, same as the author's implementation
        # v = quadprog.solve_qp(P, q, G, h)[0]
        # using new library qpsolvers
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()


class GEMPlugin(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.memory_strength = memory_strength

        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()
        self.memory_tid: Dict[int, Tensor] = dict()

        self.G: Tensor = torch.empty(0)


    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if strategy.clock.train_exp_counter > 0:
            G = []
            strategy.model.train()
            for t in range(strategy.clock.train_exp_counter):
                strategy.model.train()
                strategy.optimizer.zero_grad()
                # xref = self.memory_x[t].to(strategy.device)
                
                if isinstance(self.memory_x[t], dict):
                    # If memory_x is a dictionary, move each tensor to the device
                    xref = {k: v.to(strategy.device) for k, v in self.memory_x[t].items()}
                else:
                    # For regular tensor input
                    xref = self.memory_x[t].to(strategy.device)
                    
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward_base(strategy.model, xref, self.memory_tid[t])
                loss = strategy._criterion(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten()
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=strategy.device)
                            )
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        Handles both tensor and dictionary inputs.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        tot = 0
        
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            
            # Handle batch size calculation
            if isinstance(x, dict):
                # Take the first tensor to determine batch size
                first_key = next(iter(x))
                current_batch_size = x[first_key].shape[0]
            else:
                current_batch_size = x.shape[0]
            
            # Check if we can add the full batch
            if tot + current_batch_size <= self.patterns_per_experience:
                if t not in self.memory_x:
                    # Initialize memory for this task
                    if isinstance(x, dict):
                        # For dictionary input, create a dictionary in memory
                        self.memory_x[t] = {k: v.clone() for k, v in x.items()}
                    else:
                        self.memory_x[t] = x.clone()
                    
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    # Concat to existing memory
                    if isinstance(x, dict):
                        # For dictionary input, concatenate each tensor in the dictionary
                        for k in x.keys():
                            self.memory_x[t][k] = torch.cat((self.memory_x[t][k], x[k]), dim=0)
                    else:
                        self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)
            else:
                # We need to add only part of the batch
                diff = self.patterns_per_experience - tot
                
                if t not in self.memory_x:
                    # Initialize memory with partial batch
                    if isinstance(x, dict):
                        self.memory_x[t] = {k: v[:diff].clone() for k, v in x.items()}
                    else:
                        self.memory_x[t] = x[:diff].clone()
                    
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    # Concat partial batch to existing memory
                    if isinstance(x, dict):
                        for k in x.keys():
                            self.memory_x[t][k] = torch.cat((self.memory_x[t][k], x[k][:diff]), dim=0)
                    else:
                        self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid[:diff]), dim=0)
                break
            
            tot += current_batch_size
    # def update_memory(self, dataset, t, batch_size):
    #     """
    #     Update replay memory with patterns from current experience.
    #     """
    #     collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
    #     dataloader = DataLoader(
    #         dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    #     )
    #     tot = 0
    #     for mbatch in dataloader:
    #         x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
    #         # if tot + x.size(0) <= self.patterns_per_experience:
    #         if tot + batch_size <= self.patterns_per_experience:
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x.clone()
    #                 self.memory_y[t] = y.clone()
    #                 self.memory_tid[t] = tid.clone()
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
    #                 self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

    #         else:
    #             diff = self.patterns_per_experience - tot
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x[:diff].clone()
    #                 self.memory_y[t] = y[:diff].clone()
    #                 self.memory_tid[t] = tid[:diff].clone()
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
    #                 self.memory_tid[t] = torch.cat(
    #                     (self.memory_tid[t], tid[:diff]), dim=0
    #                 )
    #             break
    #         # tot += x.size(0)
    #         tot += batch_size

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        # solution with old quadprog library, same as the author's implementation
        # v = quadprog.solve_qp(P, q, G, h)[0]
        # using new library qpsolvers
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
    
## Code when using GEM with DrivAerNet and potentially all pointcloud benchmarks (Shapenet yes)

class GEMPluginDRIVAERNET(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.memory_strength = memory_strength

        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()
        self.memory_tid: Dict[int, Tensor] = dict()

        self.G: Tensor = torch.empty(0)
        
    def before_training_iteration(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            import time
            G = []
            batch_size = 30
            strategy.model.train()
            
            for t in range(strategy.clock.train_exp_counter):
                start_exp = time.time()
                
                # Time data loading
                data_start = time.time()
                # xref = self.memory_x[t].to(strategy.device)
                
                if isinstance(self.memory_x[t], dict):
                    # If memory_x is a dictionary, move each tensor to the device
                    xref = {k: v.to(strategy.device) for k, v in self.memory_x[t].items()}
                    # Get the batch size from the first tensor in the dictionary
                    first_key = next(iter(xref))
                    n_samples = xref[first_key].shape[0]
                else:
                    # For regular tensor input
                    xref = self.memory_x[t].to(strategy.device)
                    n_samples = len(xref)
                    
                yref = self.memory_y[t].to(strategy.device)
                # n_samples = len(xref)
                data_time = time.time() - data_start
                
                forward_start = time.time()
                all_outputs = []
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    # batch_x = xref[i:batch_end]
                    if isinstance(xref, dict):
                        # If xref is a dictionary, slice each tensor in the dictionary
                        batch_x = {k: v[i:batch_end] for k, v in xref.items()}
                    else:
                        # For regular tensor input
                        batch_x = xref[i:batch_end]
                    batch_tid = self.memory_tid[t][i:batch_end] if self.memory_tid[t] is not None else None
                    out = avalanche_forward_base(strategy.model, batch_x, batch_tid)
                    all_outputs.append(out)
                forward_time = time.time() - forward_start
                
                # Time concatenation
                cat_start = time.time()
                combined_outputs = torch.cat(all_outputs, dim=0)
                cat_time = time.time() - cat_start
                
                backward_start = time.time()
                strategy.optimizer.zero_grad()
                loss = strategy._criterion(combined_outputs, yref)
                loss.backward()
                backward_time = time.time() - backward_start
                
                # Time gradient collection
                grad_start = time.time()
                G.append(
                    torch.cat(
                        [
                            (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=strategy.device))
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )
                grad_time = time.time() - grad_start
                
                exp_time = time.time() - start_exp
                # print(f"Experience {t} breakdown:")
                # print(f"  Data loading: {data_time:.2f}s")
                # print(f"  Forward passes: {forward_time:.2f}s")
                # print(f"  Output concatenation: {cat_time:.2f}s")
                # print(f"  Backward pass: {backward_time:.2f}s")
                # print(f"  Gradient collection: {grad_time:.2f}s")
                # print(f"  Total time: {exp_time:.2f}s")

            self.G = torch.stack(G)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        Handles both tensor and dictionary inputs.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        tot = 0
        
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            
            # Handle batch size calculation
            if isinstance(x, dict):
                # Take the first tensor to determine batch size
                first_key = next(iter(x))
                current_batch_size = x[first_key].shape[0]
            else:
                current_batch_size = x.shape[0]
            
            # Check if we can add the full batch
            if tot + current_batch_size <= self.patterns_per_experience:
                if t not in self.memory_x:
                    # Initialize memory for this task
                    if isinstance(x, dict):
                        # For dictionary input, create a dictionary in memory
                        self.memory_x[t] = {k: v.clone() for k, v in x.items()}
                    else:
                        self.memory_x[t] = x.clone()
                    
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    # Concat to existing memory
                    if isinstance(x, dict):
                        # For dictionary input, concatenate each tensor in the dictionary
                        for k in x.keys():
                            self.memory_x[t][k] = torch.cat((self.memory_x[t][k], x[k]), dim=0)
                    else:
                        self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)
            else:
                # We need to add only part of the batch
                diff = self.patterns_per_experience - tot
                
                if t not in self.memory_x:
                    # Initialize memory with partial batch
                    if isinstance(x, dict):
                        self.memory_x[t] = {k: v[:diff].clone() for k, v in x.items()}
                    else:
                        self.memory_x[t] = x[:diff].clone()
                    
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    # Concat partial batch to existing memory
                    if isinstance(x, dict):
                        for k in x.keys():
                            self.memory_x[t][k] = torch.cat((self.memory_x[t][k], x[k][:diff]), dim=0)
                    else:
                        self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid[:diff]), dim=0)
                break
            
            tot += current_batch_size
    # def update_memory(self, dataset, t, batch_size):
    #     """
    #     Update replay memory with patterns from current experience.
    #     """
    #     collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
    #     dataloader = DataLoader(
    #         dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    #     )
    #     tot = 0
    #     for mbatch in dataloader:
    #         x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
    #         if tot + x.size(0) <= self.patterns_per_experience:
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x.clone()
    #                 self.memory_y[t] = y.clone()
    #                 self.memory_tid[t] = tid.clone()
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
    #                 self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

    #         else:
    #             diff = self.patterns_per_experience - tot
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x[:diff].clone()
    #                 self.memory_y[t] = y[:diff].clone()
    #                 self.memory_tid[t] = tid[:diff].clone()
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
    #                 self.memory_tid[t] = torch.cat(
    #                     (self.memory_tid[t], tid[:diff]), dim=0
    #                 )
    #             break
    #         tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        # solution with old quadprog library, same as the author's implementation
        # v = quadprog.solve_qp(P, q, G, h)[0]
        # using new library qpsolvers
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
