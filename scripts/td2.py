import jax
import jax.numpy as jnp
import flax
from flax import nnx

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    ema_model: nnx.Module
    encoder: nnx.Module
    optimizer: nnx.Optimizer

def make_flow_functions(cfg, data_shape):
    eval_dt = 1 / cfg.eval_denoise_steps
    @nnx.jit
    def sample(model: nnx.Module, encoder: nnx.Module, batch: dict, key: jnp.ndarray):
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
        return x_current_raw, encoder.decode(x_1)

    @nnx.jit
    def train_step(training_state: TrainingState, batch: dict, key: jnp.ndarray):
        x_current_raw = batch["observations"].astype(jnp.float32) / 255.0
        x_current = training_state.encoder(x_current_raw)
        a_current = batch["actions"]
        key, key_next = jax.random.split(key)
        x_next_raw = batch["next_observations"].astype(jnp.float32) / 255.0
        x_next = training_state.encoder(x_next_raw, key_next)
        a_next = batch["next_actions"]
        mask = batch["masks"][:, None, None, None]

        key, key_time = jax.random.split(key)
        t_cfm = jax.random.uniform(key_time, shape=((x_current.shape[0]),))
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x_current.shape)
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
        t_bootstrap = jax.random.uniform(key_time, shape=((x_current.shape[0]),))
        dt_bootstrap = t_bootstrap / cfg.target_denoise_steps
        key, key_noise = jax.random.split(key)
        x_0 = jax.random.normal(key_noise, x_current.shape)
        x_t_bootstrap, t_bootstrap = sample_step((x_0, jnp.zeros((x_current.shape[0], ))), dt_bootstrap)
        v_t_bootstrap = jax.lax.stop_gradient(training_state.ema_model(x_t_bootstrap, x_next, a_next, t_bootstrap))

        def loss_fn(model: nnx.Module):
            next_state_loss = jnp.mean((v_t_cfm - model(x_t_cfm, x_current, a_current, t_cfm)) ** 2) 
            bootstrap_loss = jnp.mean(
                mask * (v_t_bootstrap - model(x_t_bootstrap, x_current, a_current, t_bootstrap)) ** 2
            )

            return (1 - cfg.gamma) * next_state_loss + cfg.gamma * bootstrap_loss
        
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
