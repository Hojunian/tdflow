import argparse
import os
import sys
import time
import yaml
import matplotlib.pyplot as plt
import wandb

import jax
import jax.numpy as jnp
import optax
import flax
from flax import nnx
import orbax.checkpoint as ocp
import numpy as np

from model.encoder import Encoder
from model.network import DiT2D_GHM

from utils.config import str2bool
from utils.ogbench import make_datasets
import time

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    ema_model: nnx.Module
    encoder: nnx.Module
    ema_encoder: nnx.Module
    optimizer: nnx.Optimizer
    encoder_optimizer: nnx.Optimizer

def make_flow_functions(cfg, data_shape):
    eval_dt = 1 / cfg.eval_denoise_steps
    @nnx.jit
    def sample(model: nnx.Module, encoder: nnx.Module, batch: dict, key: jnp.ndarray):
        key, key_current = jax.random.split(key)
        x_current_raw = batch["observations"].astype(jnp.float32) / 255.0
        x_current = encoder(x_current_raw)
        a_current = batch["actions"]
        @nnx.scan(length=cfg.eval_denoise_steps, in_axes=(nnx.Carry), out_axes=(nnx.Carry))
        def sample_step(carry):
            x_t, t = carry

            x_t_mid = x_t + model(x_t, x_current, a_current, t) * (eval_dt / 2)
            t_mid = t + (eval_dt / 2)

            x_t += model(x_t_mid, x_current, a_current, t_mid) * eval_dt
            t += eval_dt
            return (x_t, t)
        
        x_t = jax.random.normal(key, shape=(cfg.eval_batch_size,) + data_shape)
        t = jnp.zeros((cfg.eval_batch_size,), dtype=jnp.float32)

        x_1 = sample_step((x_t,t))[0]

        z, dist = encoder.encode(x_current_raw, key_current)
        y_1 = encoder.decode(z)
        mu = dist.mean
        logvar = dist.logvar

        loss_mse = jnp.mean((x_current_raw - y_1) ** 2)
        loss_kl = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1)
        info = {
            "eval/loss_mse": loss_mse,
            "eval/loss_kl": loss_kl,
        }

        return x_current_raw, encoder.decode(x_1), info

    @nnx.jit
    def train_step(training_state: TrainingState, batch: dict, key: jnp.ndarray):
        key, key_current = jax.random.split(key)
        x_current_raw = batch["observations"].astype(jnp.float32) / 255.0
        a_current = batch["actions"]
        key, key_next = jax.random.split(key)
        x_next_raw = batch["next_observations"].astype(jnp.float32) / 255.0
        x_next = training_state.ema_encoder(x_next_raw, key_next)
        a_next = batch["next_actions"]
        mask = batch["masks"][:, None, None, None]

        key, key_time = jax.random.split(key)
        t_cfm = jax.random.uniform(key_time, shape=((x_next.shape[0]),))
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x_next.shape)
        x_t_cfm = x_0 + t_cfm[:, None, None, None] * (x_next - (1 - 1e-5) * x_0)
        v_t_cfm = x_next - (1 - 1e-5) * x_0

        @nnx.scan(length=cfg.target_denoise_steps, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry))
        def sample_step(carry, dt):
            x_t, t = carry

            x_t_mid = x_t + training_state.ema_model(x_t, x_next, a_next, t) * (dt[:, None, None, None] / 2)
            t_mid = t + (dt / 2)

            x_t += training_state.ema_model(x_t_mid, x_next, a_next, t_mid) * dt[:, None, None, None]
            t += dt
            return (x_t, t)

        key, key_time = jax.random.split(key)
        t_bootstrap = jax.random.uniform(key_time, shape=((x_next.shape[0]),))
        dt_bootstrap = t_bootstrap / cfg.target_denoise_steps
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x_next.shape)
        x_t_bootstrap, t_bootstrap = sample_step((x_0, jnp.zeros((x_next.shape[0], ))), dt_bootstrap)
        v_t_bootstrap = jax.lax.stop_gradient(training_state.ema_model(x_t_bootstrap, x_next, a_next, t_bootstrap))

        def loss_fn(model: nnx.Module, encoder: nnx.Module):
            x_current = encoder(x_current_raw)
            next_state_loss = jnp.mean((v_t_cfm - model(x_t_cfm, x_current, a_current, t_cfm)) ** 2) 
            bootstrap_loss = jnp.mean(
                mask * (v_t_bootstrap - model(x_t_bootstrap, x_current, a_current, t_bootstrap)) ** 2
            )

            z, dist = encoder.encode(x_current_raw, key_current)
            y_1 = encoder.decode(z)
            mu = dist.mean
            logvar = dist.logvar

            loss_fm = (1 - cfg.gamma) * next_state_loss + cfg.gamma * bootstrap_loss
            loss_mse = jnp.mean((x_current_raw - y_1) ** 2)
            loss_kl = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1)

            loss = loss_fm + cfg.encoder_weight * (loss_mse + cfg.kl_weight * loss_kl)
            info = {
                "train/loss": loss_fm,
                "train/loss_mse": loss_mse,
                "train/loss_kl": loss_kl,
            }
            return loss, info            
        
        (_, info), grads = nnx.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(
            training_state.model, training_state.encoder
        )
        training_state.optimizer.update(training_state.model, grads[0])
        training_state.encoder_optimizer.update(training_state.encoder, grads[1])

        ema_params = jax.tree_util.tree_map(
            lambda p, tp: p * (1 - cfg.ema_decay) + tp * cfg.ema_decay,
            nnx.state(training_state.model),
            nnx.state(training_state.ema_model),
        )
        nnx.update(training_state.ema_model, ema_params)

        ema_encoder_params = jax.tree_util.tree_map(
            lambda p, tp: p * (1 - cfg.ema_decay) + tp * cfg.ema_decay,
            nnx.state(training_state.encoder),
            nnx.state(training_state.ema_encoder),
        )
        nnx.update(training_state.ema_encoder, ema_encoder_params)

        return info

    return sample, train_step

def run(cfg):
    # seed setting
    np.random.seed(cfg.seed)
    # make save dir
    os.makedirs(cfg.save_dir, exist_ok=True)
    short_dataset_name = cfg.dataset_name.replace("visual-","").replace("-singletask","").replace("-v0","")
    save_dir = os.path.join(cfg.save_dir, short_dataset_name, cfg.run_name)
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    encoder_save_dir = save_dir = os.path.join(cfg.save_dir, short_dataset_name, "encoderKL2")
    encoder_save_dir = os.path.abspath(os.path.expanduser(encoder_save_dir))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cfg.yaml"), "w") as outfile:
        yaml.dump(cfg, outfile)

    if not cfg.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="tdflow",
        name=cfg.run_name,
        config=cfg,
        save_code=True,
    )

    output_dir = os.path.join(save_dir, "outputs")
    ckpt_dir = os.path.join(save_dir, "ckpts")
    encoder_ckpt_dir = os.path.join(encoder_save_dir, "ckpts")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # build dataset
    train_ds, eval_ds = make_datasets(cfg.dataset_name)
    obs_shape = train_ds["observations"][0].shape
    ## latent
    obs_shape = tuple(x//8 for x in obs_shape[:-1]) + (4,)
    action_dim = train_ds["actions"][0].shape[0]

    # build training state
    checkpointer = ocp.StandardCheckpointer()
    abstract_encoder = nnx.eval_shape(
        lambda: Encoder(
            rngs=nnx.Rngs(cfg.seed),
            from_pretrained=False,
        )
    )
    graphdef, abstract_state = nnx.split(abstract_encoder)
    encoder_state = checkpointer.restore(os.path.join(encoder_ckpt_dir, f"encoder_{cfg.encoder_ckpt}"), abstract_state)
    encoder = nnx.merge(graphdef, encoder_state)
    ema_encoder = Encoder(
        rngs=nnx.Rngs(cfg.seed),
        from_pretrained=False,
    )
    model = DiT2D_GHM(
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        img_size=obs_shape[:-1],
        in_channels=obs_shape[-1],
        action_dim=action_dim,
        rngs=nnx.Rngs(cfg.seed),
    )
    ema_model = DiT2D_GHM(
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        img_size=obs_shape[:-1],
        in_channels=obs_shape[-1],
        action_dim=action_dim,
        rngs=nnx.Rngs(cfg.seed),
    )
    nnx.update(ema_encoder, nnx.state(encoder))
    nnx.update(ema_model, nnx.state(model))
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=cfg.lr,
            b1=cfg.beta_1,
            b2=cfg.beta_2,
            weight_decay=cfg.weight_decay,
        ),
        wrt=nnx.Param,
    )
    encoder_optimizer = nnx.Optimizer(
        encoder,
        optax.adam(
            learning_rate=cfg.encoder_lr,   # TODO: parameterize
            b1=cfg.beta_1,
            b2=cfg.beta_2,
        ),
        wrt=nnx.Param,
    )
    training_state = TrainingState(model, ema_model, encoder, ema_encoder, optimizer, encoder_optimizer)

    # build sample, train fns
    sample_fn, train_step = make_flow_functions(cfg, obs_shape)
    sample_fn_chached = nnx.cached_partial(sample_fn, training_state.ema_model, training_state.ema_encoder)
    train_step_cached = nnx.cached_partial(train_step, training_state)

    # training
    loss_sum = 0.0
    key = jax.random.key(cfg.seed)

    start = time.time()
    for step in range(1, cfg.num_updates + 1):
        batch = train_ds.sample(cfg.batch_size)

        key, key_update = jax.random.split(key)
        info = train_step_cached(batch, key_update)
        loss_sum += info["train/loss_fm"]

        if step % cfg.log_every == 0:
            loss_avg = loss_sum / cfg.log_every
            training_time = time.time() - start
            print(f"step: {step} | loss: {loss_avg} | time: {training_time}")
            wandb.log(info, step=step)  # TODO: average
            loss_sum = 0
        
        if step % cfg.eval_every == 0 or step == 1:
            eval_batch = eval_ds.sample(cfg.eval_batch_size)
            x_gen_list = []
            for __ in range(cfg.eval_batch_size - 1):
                key, key_sample = jax.random.split(key)
                x_current, x_gen, info = sample_fn_chached(eval_batch, key_sample)
                x_gen_list.append(x_gen)

            x_gens = jnp.concatenate(x_gen_list, axis=2)
            x_current_np = np.array(x_current)
            x_current_np = np.clip(x_current_np, 0, 1)
            x_generated_np = np.array(x_gens)
            x_generated_np = np.clip(x_generated_np, 0, 1)
            x_np = np.concatenate([x_current_np, x_generated_np], axis=2)
            rows = []
            num_rows = int(cfg.eval_batch_size)
            for i in range(num_rows):
                row = x_np[i]
                rows.append(row)

            x_render = np.concatenate(rows, axis=0)
            plt.imshow(x_render)
            plt.savefig(os.path.join(output_dir, f"samples_{step}"))
            wandb.log({**info, "eval/samples": wandb.Image(x_render)}, step=step)

    #     if (step + 1) % config["SAVE_EVERY"] == 0:
    #         ema_algo = nnx.merge(graphdef, ema_params)
    #         ema_algo_state = nnx.state(ema_algo)
    #         checkpointer.save(os.path.join(ckpt_dir, f"ema_{step + 1}"), ema_algo_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset_name", type=str, default="visual-scene-play-singletask-v0")
    # network architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=12)
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--target_denoise_steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num_updates", type=int, default=50000)
    parser.add_argument("--encoder_weight", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1e-5)
    parser.add_argument("--encoder_ckpt", type=str, default="25000")
    # eval and logging
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_denoise_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=25000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--save_dir", type=str, default="results/"
    )
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--run_name", type=str, default="demo")

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    run(args)
