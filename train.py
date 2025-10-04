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
import ogbench

from model.encoder import Encoder, EncodedDiT2D

from utils.config import str2bool
from utils.ogbench import Dataset
import time

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    ema_model: nnx.Module
    optimizer: nnx.Optimizer

def make_flow_functions(cfg, data_shape):
    @nnx.jit
    def sample(model: nnx.Module, key: jnp.ndarray):
        @nnx.scan(length=cfg.denoise_timesteps//cfg.eval_dt, in_axes=(nnx.Carry), out_axes=(nnx.Carry))
        def sample_step(carry):
            x_t, t = carry
            dx = model(x_t, t) * (cfg.eval_dt / cfg.denoise_timesteps)
            x_t += dx
            t += cfg.eval_dt
            return (x_t, t)
        
        x_t = jax.random.normal(key, shape=(cfg.eval_batch_size,) + data_shape)
        t = jnp.zeros((cfg.eval_batch_size,), dtype=int)

        return sample_step((x_t,t))[0]

    @nnx.jit
    def train_step(training_state: TrainingState, x: jnp.ndarray, key: jnp.ndarray):
        key, key_time = jax.random.split(key)
        t = jax.random.randint(
            key_time,
            shape=((x.shape[0]),),
            minval=0,
            maxval=cfg.denoise_timesteps,
        )
        t_full = t / cfg.denoise_timesteps
        
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x.shape)
        x_t = x_0 + t_full[:, None, None, None] * (x - (1 - 1e-5) * x_0)
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
    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(cfg.dataset_name)
    for k in ['qpos', 'qvel', 'button_states']:
        if k in train_dataset:
            del train_dataset[k]
        if k in val_dataset:
            del val_dataset[k]
    train_ds = Dataset.create(**train_dataset)
    val_ds = Dataset.create(**val_dataset)
    obs_shape = train_ds["observations"][0].shape

    checkpointer = ocp.StandardCheckpointer()

    # build training state
    abstract_encoder = nnx.eval_shape(
        lambda: Encoder(
            rngs=nnx.Rngs(cfg.seed),
            from_pretrained=False,
        )
    )
    graphdef, abstract_state = nnx.split(abstract_encoder)
    state = checkpointer.restore(os.path.join(encoder_ckpt_dir, f"encoder_{15000}"), abstract_state)
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
    nnx.update(model.encoder, state)
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


        if (step + 1) % cfg.save_every == 0:
            _, state = nnx.split(ema_model)
            checkpointer.save(os.path.join(ckpt_dir, f"ema_{step + 1}"), state)

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
    parser.add_argument("--data_type", type=str, default="image")
    # network architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=12)
    # flow matching
    parser.add_argument("--algo", type=str, default="flow_matching")
    parser.add_argument("--denoise_timesteps", type=int, default=128)
    parser.add_argument("--denoise_path", type=str, default="cond_ot")
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
    # eval and logging
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_dt", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=25000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--save_dir", type=str, default="results/"
    )
    parser.add_argument("--run_name", type=str, default="demo")

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    run(args)
