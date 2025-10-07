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

from model.encoder import Encoder, EncodedDiT2D

from utils.config import str2bool
from utils.ogbench import make_datasets
import time

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    ema_model: nnx.Module
    optimizer: nnx.Optimizer

def make_flow_functions(cfg, data_shape):
    eval_dt = 1 / cfg.eval_denoise_steps
    @nnx.jit
    def sample(model: nnx.Module, key: jnp.ndarray):
        @nnx.scan(length=cfg.eval_denoise_steps, in_axes=(nnx.Carry), out_axes=(nnx.Carry))
        def sample_step(carry):
            x_t, t = carry

            x_t_mid = x_t + model(x_t, t) * (eval_dt / 2)
            t_mid = t + (eval_dt / 2)

            x_t += model(x_t_mid, t_mid) * eval_dt
            t += eval_dt
            return (x_t, t)
        
        x_t = jax.random.normal(key, shape=(cfg.eval_batch_size,) + data_shape)
        t = jnp.zeros((cfg.eval_batch_size,), dtype=jnp.float32)

        x_1 = sample_step((x_t,t))[0]
        return model.encoder.decode(x_1)

    @nnx.jit
    def train_step(training_state: TrainingState, x: jnp.ndarray, key: jnp.ndarray):
        key, key_encoder = jax.random.split(key)
        x = training_state.model.encoder(x, key_encoder)
        
        key, key_time = jax.random.split(key)
        t = jax.random.uniform(key_time, shape=((x.shape[0]),))
        
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x.shape)
        x_t = x_0 + t[:, None, None, None] * (x - (1 - 1e-5) * x_0)
        v_t = x - (1 - 1e-5) * x_0

        def loss_fn(model: nnx.Module):
            v_t_pred = model(x_t, t)
            return jnp.mean((v_t - v_t_pred) ** 2)
        
        loss, grads = nnx.value_and_grad(loss_fn)(training_state.model)
        training_state.optimizer.update(training_state.model, grads)

        ema_params = jax.tree_util.tree_map(
            lambda p, tp: p * (1 - cfg.ema_decay) + tp * cfg.ema_decay,
            nnx.state(training_state.model),
            nnx.state(training_state.ema_model),
        )
        nnx.update(training_state.ema_model, ema_params)

        return loss

    return sample, train_step

def run(cfg):
    # seed setting
    np.random.seed(cfg.seed)
    # make save dir
    os.makedirs(cfg.save_dir, exist_ok=True)
    short_dataset_name = cfg.dataset_name.replace("visual-","").replace("-singletask","").replace("-v0","")
    save_dir = os.path.join(cfg.save_dir, short_dataset_name, cfg.run_name)
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    encoder_save_dir = save_dir = os.path.join(cfg.save_dir, short_dataset_name, "encoder")
    encoder_save_dir = os.path.abspath(os.path.expanduser(encoder_save_dir))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cfg.yaml"), "w") as outfile:
        yaml.dump(cfg, outfile)

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
    train_ds, __ = make_datasets(cfg.dataset_name)
    obs_shape = train_ds["observations"][0].shape
    ## latent
    obs_shape = tuple(x//8 for x in obs_shape[:-1]) + (4,)

    # build training state
    checkpointer = ocp.StandardCheckpointer()
    abstract_encoder = nnx.eval_shape(
        lambda: Encoder(
            rngs=nnx.Rngs(cfg.seed),
            from_pretrained=False,
        )
    )
    _, abstract_state = nnx.split(abstract_encoder)
    encoder_state = checkpointer.restore(os.path.join(encoder_ckpt_dir, f"encoder_{cfg.encoder_ckpt}"), abstract_state)
    model = EncodedDiT2D(
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        img_size=obs_shape[:-1],
        in_channels=obs_shape[-1],
        action_dim=1,
        rngs=nnx.Rngs(cfg.seed),
    )
    ema_model = EncodedDiT2D(
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        img_size=obs_shape[:-1],
        in_channels=obs_shape[-1],
        action_dim=1,
        rngs=nnx.Rngs(cfg.seed),
    )
    nnx.update(model.encoder, encoder_state)
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
    training_state = TrainingState(model, ema_model, optimizer)

    # build sample, train fns
    sample_fn, train_step = make_flow_functions(cfg, obs_shape)
    sample_fn_chached = nnx.cached_partial(sample_fn, training_state.model)
    train_step_cached = nnx.cached_partial(train_step, training_state)

    # setup train, sample function
    metric = nnx.metrics.Average()

    # training
    loss_sum = 0.0
    key = jax.random.key(cfg.seed)

    start = time.time()
    for step in range(1, cfg.num_updates + 1):
        batch = train_ds.sample(cfg.batch_size)
        x_1 = batch["observations"].astype(jnp.float32) / 255.0

        key, key_update = jax.random.split(key)
        loss_sum += train_step_cached(x_1, key_update)

        if step % cfg.log_every == 0:
            loss_avg = loss_sum / cfg.log_every
            training_time = time.time() - start
            print(f"step: {step} | loss: {loss_avg} | time: {training_time}")
            wandb.log(
                {
                    "train/loss": loss_avg,
                },
                step=step,
            )
            loss_sum = 0

        if step % cfg.eval_every == 0 or step == 1:
            key, key_sample = jax.random.split(key)
            x_generated = sample_fn_chached(key_sample)
            
            x_np = np.array(x_generated)
            x_np = np.clip(x_np, 0, 1)

            rows = []
            num_rows = int(cfg.eval_batch_size ** 0.5)
            for i in range(num_rows):
                row = np.concatenate([x_np[i * num_rows + j] for j in range(4)], axis=1)
                rows.append(row)

            x_render = np.concatenate(rows, axis=0)
            plt.imshow(x_render)
            plt.savefig(os.path.join(output_dir, f"samples_{step}"))
            wandb.log({"eval/samples": wandb.Image(x_render)}, step=step)


    #     if (step + 1) % config["SAVE_EVERY"] == 0:
    #         ema_algo = nnx.merge(graphdef, ema_params)
    #         ema_algo_state = nnx.state(ema_algo)
    #         checkpointer.save(os.path.join(ckpt_dir, f"ema_{step + 1}"), ema_algo_state)

    checkpointer.wait_until_finished()
    checkpointer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset_name", type=str, default="visual-scene-play-singletask-v0")
    # network architecture
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--num_updates", type=int, default=100000)
    parser.add_argument("--encoder_ckpt", type=str, default="25000")
    # eval and logging
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_denoise_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=25000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--save_dir", type=str, default="results/"
    )
    parser.add_argument("--run_name", type=str, default="demo")

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    run(args)
