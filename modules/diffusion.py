import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import generative_models.sgm.inference.helpers as helpers

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from eval.reconstruction_eval import ReconstructionEval

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.modules.ema import LitEma
from sgm.util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)

from modules.lit_ema_cpu import LitEmaCPU


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        ema_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        ema_update_every: int = 100,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        # if self.use_ema:
        self.model_ema = LitEmaCPU(
            self.model,
            decay=ema_decay_rate,
            update_every=ema_update_every,
        )
            
            # (
            #     instantiate_from_config(ema_config)
            #     if ema_config is not None
            #     else LitEma(self.model, decay=ema_decay_rate)
            # )
            
        print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        # self.save_hyperparameters()

        self.validation_step_outputs = []

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]
    
    def set_wandb(self, wandb):
        self.wandb = wandb

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)

        # if eval_recons:
        #     c, uc = self.conditioner.get_unconditional_conditioning(
        #         batch,
        #         force_uc_zero_embeddings=["ltnt"]
        #         if len(self.conditioner.embedders) > 0
        #         else [],
        #     )

        #     N = 8

        #     z = x.to(self.device)[:N]

        #     for k in c:
        #         if isinstance(c[k], torch.Tensor):
        #             c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        #     with self.ema_scope("Plotting"):
        #         samples = self.sample(
        #             c, shape=z.shape[1:], uc=uc, batch_size=N
        #         )
            
        #     recon_evaluator = ReconstructionEval(self.device)
        #     recon_evaluator.update_fid(self.get_input(batch), samples)
        #     evals = recon_evaluator(samples, self.get_input(batch))

        #     return loss, loss_dict, evals
            
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # self.wandb.log({
        #     "train/loss": loss,
        #     "global_step": self.global_step
        # }, step=self.global_step, commit=True)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict = self.shared_step(batch)

            self.validation_step_outputs.append(loss.cpu())

            # self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            # self.wandb.log({
            #     "val/loss": loss,
            # })
            # if self.scheduler_config is not None:
            #     lr = self.optimizers().param_groups[0]["lr"]
            #     self.log(
            #         "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            #     )

            if batch_idx == 0:
                with self.ema_scope():
                    imgs_log = self.log_recons_images(batch, N=2, ucg_keys=["ltnt"])
                    self.logger.log_image(
                        key="Original",
                        images=[img for img in imgs_log["inputs"].cpu().detach().float()]
                    )
                    self.logger.log_image(
                        key="Reconstructions",
                        images=[img for img in imgs_log["samples"].cpu().detach().float()]
                    )

        return loss
    
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        
        # self.wandb.log({
        #     "val/loss": loss,
        # }, step=self.global_step, commit=False)
        self.validation_step_outputs.clear()  # free memory

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt
    
    @torch.no_grad()
    def log_recons_images(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        
        samples = self.sample(
            c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
        )
        # samples = self.decode_fi
        log["samples"] = samples
        return log
    
    @torch.no_grad()
    def generate(self, cond: Dict, batch_size, ucg_keys: List[str] = None, shape=None, **kwargs):
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        embs = helpers.get_unique_embedder_keys_from_conditioner(self.conditioner)
        batch, batch_uc = helpers.get_batch(
            embs, 
            cond,
            batch_size,
        )

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        print(x)

        sampling_kwargs = {}

        # x = x.to(self.device)[:batch_size]
        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:batch_size].to(self.device), (c, uc))

        
        samples = self.sample(
            c, shape=shape, uc=uc, batch_size=batch_size, **sampling_kwargs
        )

        decoded_latent = self.decode_first_stage(samples)

        return decoded_latent


    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    # @torch.no_grad()
    # def log_images(
    #     self,
    #     batch: Dict,
    #     N: int = 8,
    #     sample: bool = True,
    #     ucg_keys: List[str] = None,
    #     **kwargs,
    # ) -> Dict:
    #     conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
    #     if ucg_keys:
    #         assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
    #             "Each.logger defined ucg key for sampling must be in the provided conditioner input keys,"
    #             f"but we have {ucg_keys} vs. {conditioner_input_keys}"
    #         )
    #     else:
    #         ucg_keys = conditioner_input_keys
    #     log = dict()

    #     x = self.get_input(batch)

    #     c, uc = self.conditioner.get_unconditional_conditioning(
    #         batch,
    #         force_uc_zero_embeddings=ucg_keys
    #         if len(self.conditioner.embedders) > 0
    #         else [],
    #     )

    #     sampling_kwargs = {}

    #     N = min(x.shape[0], N)
    #     x = x.to(self.device)[:N]
    #     log["inputs"] = x
    #     z = self.encode_first_stage(x)
    #     log["reconstructions"] = self.decode_first_stage(z)
    #     log.update(self.log_conditionings(batch, N))

    #     for k in c:
    #         if isinstance(c[k], torch.Tensor):
    #             c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

    #     if sample:
    #         with self.ema_scope("Plotting"):
    #             samples = self.sample(
    #                 c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
    #             )
    #         samples = self.decode_first_stage(samples)
    #         log["samples"] = samples
    #     return log
