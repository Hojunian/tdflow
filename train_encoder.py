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

from utils.config import str2bool
from utils.ogbench import make_datasets
import time

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    optimizer: nnx.Optimizer

def make_flow_functions(cfg):
    @nnx.jit
    def train_step(training_state, x_1, key):
        def loss_fn(model: nnx.Module):
            mu, logvar = model.encode(x_1)
            noise = jax.random.normal(key, shape=mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * noise
            y_1 = model.decode(z)

            loss_mse = jnp.mean((x_1 - y_1) ** 2)
            loss_l1 = jnp.mean(jnp.abs(x_1 - y_1))
            loss_kl = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1)
            
            if cfg.loss_type == 'l1':
                return loss_l1 + cfg.kl_weight * loss_kl
            else:
                return loss_mse + cfg.kl_weight * loss_kl
            
        loss, grads = nnx.value_and_grad(loss_fn)(training_state.model)
        training_state.optimizer.update(training_state.model, grads)
        return loss

    @nnx.jit
    def val_step(model, x_1, key):
        mu, logvar = model.encode(x_1)
        noise = jax.random.normal(key, shape=mu.shape)
        z = mu + jnp.exp(0.5 * logvar) * noise
        y_1 = model.decode(z)
        
        loss_mse = jnp.mean((x_1 - y_1) ** 2)
        loss_l1 = jnp.mean(jnp.abs(x_1 - y_1))
        loss_kl = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1)


        return loss_mse, loss_l1, loss_kl, model.decode(mu)

    return train_step, val_step

def run(cfg):
    # seed setting
    np.random.seed(cfg.seed)
    # make save dir
    os.makedirs(cfg.save_dir, exist_ok=True)
    short_dataset_name = cfg.dataset_name.replace("visual-","").replace("-singletask","").replace("-v0","")
    save_dir = os.path.join(cfg.save_dir, short_dataset_name, cfg.run_name)
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
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
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # build dataset
    train_ds, val_ds = make_datasets(cfg.dataset_name)
    obs_shape = train_ds["observations"][0].shape

    # build training state
    checkpointer = ocp.StandardCheckpointer()
    encoder = Encoder(
        rngs=nnx.Rngs(cfg.seed),
        img_size=obs_shape[:-1],
        in_features=obs_shape[-1],
    )
    optimizer = nnx.Optimizer(
        encoder,
        optax.adamw(
            learning_rate=cfg.lr,
            b1=cfg.beta_1,
            b2=cfg.beta_2,
            weight_decay=cfg.weight_decay,
        ),
        wrt=nnx.Param,
    )
    training_state = TrainingState(encoder, optimizer)

    # build sample, train fns
    train_step, val_step = make_flow_functions(cfg)
    train_step_cached = nnx.cached_partial(train_step, training_state)
    val_step_cached = nnx.cached_partial(val_step, training_state.model)

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
            batch = val_ds.sample(cfg.eval_batch_size)
            x_val = batch["observations"].astype(jnp.float32) / 255.0
            
            key, key_val = jax.random.split(key)
            loss_mse, loss_l1, loss_kl, y_val = val_step_cached(x_val, key_val)

            x_np = np.array(x_val)
            y_np = np.array(y_val)
            y_np = np.clip(y_np, 0, 1)
            
            rows = []
            for i in range(cfg.eval_batch_size):
                row = np.concatenate([x_np[i], y_np[i]], axis=1)
                rows.append(row)
            x_render = np.concatenate(rows, axis=0)
            plt.imshow(x_render)
            plt.savefig(os.path.join(output_dir, f"samples_{step}"))
            wandb.log({
                    "eval/loss_mse": loss_mse,
                    "eval/loss_l1": loss_l1,
                    "eval/loss_kl": loss_kl,
                    "eval/samples": wandb.Image(x_render),
                }, 
                step=step
            )

        if (step + 1) % cfg.save_every == 0:
            _, state = nnx.split(encoder)
            checkpointer.save(os.path.join(ckpt_dir, f"encoder_{step + 1}"), state)

    checkpointer.wait_until_finished()
    checkpointer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=str2bool, default=False)
    # dataset
    """
    always put channel dimension at the end
    """
    parser.add_argument("--dataset_name", type=str, default="visual-scene-play-singletask-v0")
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_updates", type=int, default=25000)
    parser.add_argument("--kl_weight", type=float, default=1e-5)
    parser.add_argument("--loss_type", type=str, default="l2")
    # eval and logging
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--save_dir", type=str, default="results/"
    )
    parser.add_argument("--run_name", type=str, default="encoderKL2_impala")

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    run(args)
