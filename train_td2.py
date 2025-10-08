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
from flax import nnx
import orbax.checkpoint as ocp
import numpy as np

from model.encoder import Encoder
from model.network import DiT2D_GHM

from utils.config import str2bool
from utils.ogbench import make_datasets
from scripts.td2 import TrainingState, make_flow_functions
import time

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
    training_state = TrainingState(model, ema_model, encoder, optimizer)

    # build sample, train fns
    sample_fn, train_step = make_flow_functions(cfg, obs_shape)
    sample_fn_chached = nnx.cached_partial(sample_fn, training_state.ema_model, training_state.encoder)
    train_step_cached = nnx.cached_partial(train_step, training_state)

    # training
    loss_sum = 0.0
    key = jax.random.key(cfg.seed)

    start = time.time()
    for step in range(1, cfg.num_updates + 1):
        batch = train_ds.sample(cfg.batch_size)

        key, key_update = jax.random.split(key)
        loss_sum += train_step_cached(batch, key_update)

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
            eval_batch = eval_ds.sample(cfg.eval_batch_size)
            x_gen_list = []
            for __ in range(cfg.eval_batch_size - 1):
                key, key_sample = jax.random.split(key)
                x_current, x_gen = sample_fn_chached(eval_batch, key_sample)
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
            wandb.log({"eval/samples": wandb.Image(x_render)}, step=step)

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
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--target_denoise_steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num_updates", type=int, default=200000)
    parser.add_argument("--encoder_ckpt", type=str, default="25000")
    # eval and logging
    parser.add_argument("--eval_every", type=int, default=10000)
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
