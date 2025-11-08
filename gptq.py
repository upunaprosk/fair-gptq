import math
import os
import sys
import time
from threading import Lock
from typing import Optional
import json, os, io
import torch
import torch.nn as nn
import transformers

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.torch import torch_empty_cache, torch_sync
from gptqmodel.quantization.quantizer import HF_OPTIMUM, Quantizer

log = setup_logger()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")

lock = Lock()  # guard against => RuntimeError: lazy wrapper should be called at most once


class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):

        layer_index = -1

        full_name = ""
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
            layer_index = module.layer_index

            full_name = module.full_name
        else:
            self.name = HF_OPTIMUM
            self.module = module

        self.layer_index = layer_index
        self.full_name = full_name
        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg
        self.device = self.module.weight.device
        self.module_copy = self._clone_module()
        self.alpha = self.qcfg.alpha
        self.rows, self.columns = self.module_copy.shape[0], self.module_copy.shape[1]
        # self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0
        self.log_delta_w = self.qcfg.log_delta_w
        self.log_hessian_sensitivity = self.qcfg.log_hessian_sensitivity
        self.sum_hessians = int(self.qcfg.sum_hessians)

        self.quantizer = self.create_quantizer(name=self.name)
        ignored_bias = False
        if self.qcfg.ignore_bias:
            for name_unbias in self.qcfg.ignore_bias.split(","):
                if name_unbias in self.full_name:
                    self.alpha = 0
                    ignored_bias = True
        found_layer = False
        if self.qcfg.select_layer:
            for layer_index_i in self.qcfg.select_layer.split(","):
                if int(self.layer_index) == int(layer_index_i):
                    if self.alpha != 0 and ignored_bias == False:
                        found_layer = True
                        self.alpha = self.qcfg.alpha

            if found_layer == False:  # if layer is not in selected
                self.alpha = 0
        if self.alpha:
            log.info(f"Selected Layer for debiasing: {self.full_name}")
            if self.sum_hessians:
                log.info(f"Using H2inv hessian for deltaW.")
        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []
        # fwd counter
        self.fwd_counter = 0

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def _clone_module(self):
        clone = self.module.weight.data.clone()

        if isinstance(self.module, nn.Conv2d):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        return clone.float()

    def add_batch(self, inp, out):
        self.fwd_counter += 1

        if self.fwd_inputs_buffered:
            self.fwd_inputs_buffered_data.append(inp.to(device=CPU))
        else:
            self.process_batch(inp)

    def process_batch(self, inp):

        inp = inp.to(device=self.device)
        # [DEBIASING CHANGE] (Fair-GPTQ)
        inp1 = inp.clone()
        if len(inp1.shape) == 2:
            inp1 = inp1.unsqueeze(
                0)  # Perhaps, we don't need this: Convert to 3D tensor [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = inp1.shape
        if batch_size != 2:
            real_seq_len = seq_len // 2
            inp1 = inp1.reshape(2, real_seq_len, hidden_dim)
            # raise ValueError("Batch size must be exactly 2 to compute debiasing term.")
        tmp = inp1.shape[0]  # batch size
        X0 = inp1[0]  # Shape: [seq_len, hidden_dim]
        X1 = inp1[1]  # Shape: [seq_len, hidden_dim]
        X0 = X0.t()  # Shape: [hidden_dim, seq_len]
        X1 = X1.t()  # Shape: [hidden_dim, seq_len]

        if not hasattr(self, "H_x01"):
            try:
                self.H_x01 = torch.zeros((self.columns, self.columns), device=self.device)
            except torch.OutOfMemoryError:
                log.info("Memory: OOM H allocate bypass")
                torch_empty_cache()
                self.H_x01 = torch.zeros((self.columns, self.columns), device=self.device)

        else:
            self.H_x01 *= self.nsamples / (self.nsamples + tmp)

        samples_Hx01 = (self.nsamples + tmp) / 2
        delta = math.sqrt(2 / samples_Hx01) * (X0 - X1).float()
        try:
            self.H_x01 += self.alpha * delta.matmul(delta.t())
        except torch.OutOfMemoryError:
            log.info("Memory: OOM cpu bypass for process batch matmul")
            torch_empty_cache()
            device = self.H_x01.device
            self.H_x01, delta = self.H_x01.to(device=CPU), delta.to(device=CPU)
            self.H_x01 += self.alpha * delta.matmul(delta.t())
            self.H_x01 = self.H_x01.to(device=device)  # move back
        # [ORIGINAL CODE] # GPTQ, for computing H_acc
        # if os.environ.get("DEBUG"):
        #     self.inp1 = inp
        #     self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if 'fc' in self.full_name.lower():
            if inp.shape[1] % 2 == 0:
                batch_size = 2

        if isinstance(self.module, nn.Linear) or isinstance(self.module, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.module, nn.Conv2d):
            unfold = nn.Unfold(
                self.module.kernel_size,
                dilation=self.module.dilation,
                padding=self.module.padding,
                stride=self.module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        if not hasattr(self, "H"):
            try:
                self.H = torch.zeros((self.columns, self.columns), device=self.device)
            except torch.OutOfMemoryError:
                log.info("Memory: OOM H allocate bypass")
                torch_empty_cache()
                self.H = torch.zeros((self.columns, self.columns), device=self.device)
        else:
            self.H *= self.nsamples / (self.nsamples + batch_size)

        self.nsamples += batch_size
        # inp = inp.float()

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        # self.H += self.chunked_matmul_t_and_t_transposed(inp, chunk_size=1024)
        try:
            self.H += inp.matmul(inp.t())
        except torch.OutOfMemoryError:
            log.info("Memory: OOM cpu bypass for process batch matmul")
            torch_empty_cache()
            device = self.H.device
            self.H, inp = self.H.to(device=CPU), inp.to(device=CPU)
            self.H += inp.matmul(inp.t())
            self.H = self.H.to(device=device)  # move back

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def block_cholesky_inverse(self, L: torch.Tensor, upper=False, block_size=512):
        """
        Optimized Cholesky inverse with O(block_size^2) memory usage.

        Args:
            L (torch.Tensor): Cholesky factor (lower triangular)
            upper (bool): If True, L is upper triangular
            block_size (int): Processing block size (tunes memory/performance)

        Returns:
            torch.Tensor: The inverse matrix
        """
        assert L.dim() == 2 and L.size(0) == L.size(1), "Input must be square"
        n = L.size(0)
        device = L.device
        dtype = L.dtype

        if upper:
            L = L.t()

        invA = torch.zeros_like(L)
        num_blocks = math.ceil(n / block_size)

        # Cache for invL blocks to avoid recomputation
        invL_cache = {}

        for k in reversed(range(num_blocks)):
            k_start = k * block_size
            k_end = min((k + 1) * block_size, n)
            k_size = k_end - k_start

            # Diagonal block inversion
            L_block = L[k_start:k_end, k_start:k_end]
            invL_block = torch.linalg.solve_triangular(
                L_block,
                torch.eye(k_size, device=device, dtype=dtype),
                upper=False
            )
            invL_cache[k] = invL_block

            # Diagonal block contribution
            invA[k_start:k_end, k_start:k_end] = invL_block.t() @ invL_block

            # Process off-diagonal blocks in parallel where possible
            for j in range(k):
                j_start = j * block_size
                j_end = min((j + 1) * block_size, n)
                j_size = j_end - j_start

                # Compute all required invL_ik blocks first
                invL_ik_blocks = []
                for i in range(k, num_blocks):
                    i_start = i * block_size
                    i_end = min((i + 1) * block_size, n)

                    if i == k:
                        invL_ik = invL_block
                    else:
                        if i in invL_cache:
                            invL_ii = invL_cache[i]
                        else:
                            L_ii = L[i_start:i_end, i_start:i_end]
                            invL_ii = torch.linalg.solve_triangular(
                                L_ii,
                                torch.eye(i_end - i_start, device=device, dtype=dtype),
                                upper=False
                            )
                            invL_cache[i] = invL_ii

                        L_ik = L[i_start:i_end, k_start:k_end]
                        invL_ik = -invL_ii @ (L_ik @ invL_block)
                        del invL_ii

                    invL_ik_blocks.append(invL_ik)
                    del invL_ik

                # Stack blocks for batched operations
                L_jk = L[j_start:j_end, k_start:k_end]

                # Compute contributions in a more vectorized way
                temp_col = torch.cat([
                    (invL_ik.t() @ L_jk.t()) for invL_ik in invL_ik_blocks
                ], dim=0)

                del invL_ik_blocks

                # Accumulate to output
                invA[j_start:j_end, k_start:k_end] = temp_col[:j_size].t()
                invA[k_start:k_end, j_start:j_end] = temp_col[:j_size]

                del temp_col

        del invL_cache
        return invA

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # process buffered inputs
        for inp in self.fwd_inputs_buffered_data:
            self.process_batch(inp)

        # release buffer
        del self.fwd_inputs_buffered_data

        # if self.device.type not in ["mps", "cpu"]:
        #     self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            W = self._clone_module()
        else:
            W = self.module_copy
            self.module_copy = None

        self.quantizer.find_params(W, weight=True)

        H = self.H
        dead = torch.diag(H) == 0
        if self.sum_hessians:
            H_2 = H.clone() + self.H_x01.clone()
        H[dead, dead] = 1
        W[:, dead] = 0
        if sum(dead) > 0:
            log.warn(f'Dead columns: {sum(dead)}')
            file_name = f"./logs/dead_columns.txt"
            with open(file_name, "a") as f:
                f.write(f'Dead columns: {sum(dead)}')
        H_x01 = self.H_x01


        del self.H
        del self.H_x01

        scale = []
        zero = []
        now_idx = 1

        if self.qcfg.static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + self.qcfg.group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if self.qcfg.desc_act:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            H_x01 = H_x01[perm][:, perm]
            if self.sum_hessians:
                H_2 = H_2[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp_percent = self.qcfg.damp_percent
        while 1 > damp_percent > 0:
            try:
                damp = damp_percent * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.device)
                H[diag, diag] += damp
                with lock:
                    H = torch.linalg.cholesky(H)
                    try:
                        H = self.block_cholesky_inverse(H, block_size=self.columns)
                    except torch.OutOfMemoryError:
                        # half the block size will use ~18% less memory but at higher accuracy loss: 1^-2 vs 1^-8
                        # worth the tradeoff since it's either oom or slightly higher accuracy loss
                        H = self.block_cholesky_inverse(H, block_size=self.columns // 2)
                        log.warn(
                            "Quantization: OOM bypassed via low memory math at a cost of lower accuracy: `cholesky_inverse`")
                    H = torch.linalg.cholesky(H, upper=True)
                    Hinv = H
                break
            except torch._C._LinAlgError as e:
                if self.qcfg.damp_auto_increment != 0:
                    log.warn(
                        f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                    damp_percent += self.qcfg.damp_auto_increment
                else:
                    log.warn(
                        "Quantization: Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`")
                    raise e
        if not (0 < damp_percent < 1):
            raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")
        if self.alpha:
            if not self.sum_hessians:
                if self.log_delta_w:
                    W_old = W.clone()
                tmp = H_x01.matmul(W.transpose(0, 1))  # result shape same as of W^t
                tmp2 = Hinv.transpose(0, 1).matmul(Hinv.matmul(tmp))
                W -= tmp2.transpose(0, 1)
                if self.log_delta_w:
                    # print("try to log log delta_w")
                    delta_W = W - W_old
                    amplitude = torch.norm(delta_W, p=2)
                    amp_rel = amplitude / (torch.norm(W_old) + 1e-12).item()
                    amplitude = amplitude.item()
                    with open('./logs/delta_w.txt', 'a') as f:
                        f.write(f'{self.full_name} abs={amplitude:.6g} rel={amp_rel:.6g}\n')
            else:
                damp_percent = self.qcfg.damp_percent
                diag = torch.arange(self.columns, device=H.device)
                mean = torch.mean(torch.diag(H_2))
                while 1 > damp_percent > 0:
                    try:
                        damp = damp_percent * mean

                        with lock:
                            # print(f"H SHAPE: {H.shape}")
                            H_2 = H_2.clone()
                            H_2[diag, diag] += damp
                            H_2 = torch.linalg.cholesky(H_2)

                            try:
                                H_2 = self.block_cholesky_inverse(H_2, block_size=self.columns)
                            except torch.OutOfMemoryError:
                                # half the block size will use ~18% less memory but at higher accuracy loss: 1^-2 vs 1^-8
                                # worth the tradeoff since it's either oom or slightly higher accuracy loss
                                H_2 = self.block_cholesky_inverse(H_2, block_size=self.columns // 2)
                                log.warn(
                                    "Quantization: OOM bypassed via low memory math at a cost of lower accuracy: `cholesky_inverse`")

                            H_2 = torch.linalg.cholesky(H_2, upper=True)
                            H2inv = H_2.float()
                            del H_2
                        break
                    except torch._C._LinAlgError as e:
                        if self.qcfg.damp_auto_increment != 0:
                            log.warn(
                                f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                            damp_percent += self.qcfg.damp_auto_increment
                        else:
                            log.warn(
                                "Quantization: Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`")
                            raise e
                if self.log_delta_w:
                    W_old = W.clone()
                tmp = H_x01.matmul(W.transpose(0, 1))  # result shape same as of W^t
                tmp2 = H2inv.transpose(0, 1).matmul(H2inv.matmul(tmp))
                W -= tmp2.transpose(0, 1)
                if self.log_delta_w:
                    delta_W = W - W_old
                    amplitude = torch.norm(delta_W, p=2)
                    amp_rel = amplitude / (torch.norm(W_old) + 1e-12).item()
                    amplitude = amplitude.item()
                    with open('./logs/delta_w.txt', 'a') as f:
                        f.write(f'{self.full_name} abs={amplitude:.6g} rel={amp_rel:.6g}\n')
        if self.log_hessian_sensitivity:
            sensitivities_H = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            Hcopy1 = H_x01[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                dcopy = Hcopy1[i, i]

                cur_idx = i1 + i

                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        if (i1 + i) % self.qcfg.group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i): (i1 + i + self.qcfg.group_size)], weight=True)

                        if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if self.qcfg.desc_act:
                            idx = perm[idx]
                        self.quantizer = groups[idx // self.qcfg.group_size]
                        cur_idx = idx

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
                norm_diff = torch.norm(W1[:, i] - Q1[:, i], p=2)
                if self.log_hessian_sensitivity:
                    if self.alpha:
                        d_times_norm = dcopy * norm_diff
                    else:
                        d_times_norm = d * norm_diff
                    # Append column-level sensitivity value with column index measured with H
                    sensitivities_H.append((cur_idx, d_times_norm.item()))

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if os.environ.get("DEBUG"):
            #     self.layer.weight.data[:, :i2] = Q[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #
            #     logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     logger.debug(torch.sum(Losses))

        torch_sync(self.device)
        avg_loss = torch.sum(Losses).item() / self.nsamples
        if math.isnan(avg_loss):
            print("Losses sum item:", torch.sum(Losses).item())
            raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if self.qcfg.desc_act:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()
        # if Q.shape != self.module.weight.shape:
        #     self.module.weight.data = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        # else:
        #     self.module.weight.data = Q.type_as(self.module.weight.data)
        #
        # # move back to self.dev
        # self.module.weight.data = self.module.weight.data.to(device=self.device)
        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        else:
            Q = Q.type_as(self.module.weight.data)
        Q = Q.to(device=self.device)

        # if os.environ.get("DEBUG"):
        #     logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        if self.log_hessian_sensitivity:
            sensitivities_H.sort(key=lambda x: x[1], reverse=True)
            try:
                all_indices = [idx for idx, _ in sensitivities_H]
                unique_indices = set(all_indices)
                if len(all_indices) != self.columns:
                    raise ValueError(f"Expected {self.columns} sensitivities_H, got {len(all_indices)}")
                if len(unique_indices) != self.columns:
                    dupes = [x for x in all_indices if all_indices.count(x) > 1]
                    raise ValueError(f"Duplicate column indices found in sensitivities_H: {set(dupes)}")
            except ValueError as e:
                log.info("sensitivities_H check failed:", str(e))
                raise  # re-raise to halt execution
            else:
                log.info("sensitivities_H index check passed. Proceeding to save.")
            file_name = f"./logs/{self.full_name}_{self.alpha}_sensitivity_H.txt"
            with open(file_name, "w") as f:
                for idx, val in sensitivities_H:
                    f.write(f"Column {idx} {val}\n")
        duration = time.time() - start
        return Q, scale, zero, g_idx, duration, avg_loss, damp_percent, self.nsamples

    def free(self):
        # if os.environ.get("DEBUG"):
        #     self.inp1 = None
        #     self.out1 = None
        #     del self.inp1
        #     del self.out1

        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        del self.module_copy
        del self.module
        if hasattr(self, "H_x01"):
            del self.H_x01

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]