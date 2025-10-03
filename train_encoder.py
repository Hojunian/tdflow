import jax
from jax import numpy as jnp
from flax import nnx
import optax
import flax
import ogbench
from model.encoder import Encoder
from utils.ogbench import Dataset

dataset_name = "visual-scene-play-singletask-v0"
batch_size = 32
seed = 0
log_every = 2

@flax.struct.dataclass
class TrainingState:
    model: nnx.Module
    optimizer: nnx.Optimizer

@nnx.jit
def train_step(training_state, x_1, key):
    def loss_fn(model: nnx.Module):
        z, dist = model.encode(x_1, key)
        y_1 = model.decode(z)
        mu = dist.mean
        logvar = dist.logvar

        loss_mse = jnp.mean((x_1 - y_1) ** 2)
        loss_kl = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1)
        
        return loss_mse + 1e-6 * loss_kl
        
    loss, grads = nnx.value_and_grad(loss_fn)(training_state.model)
    training_state.optimizer.update(training_state.model, grads)
    return loss

@nnx.jit
def val_step(model, x_1, key):
    z = model(x_1, key)
    y_1 = model.decode(z)
    return jnp.mean((x_1 - y_1) ** 2)
    
encoder = Encoder()
optimizer = nnx.Optimizer(
    encoder,
    optax.adamw(
        learning_rate=1e-4
    ),
    wrt=nnx.Param,
)
training_state = TrainingState(encoder, optimizer)
train_step_cached = nnx.cached_partial(train_step, training_state)
val_step_cached = nnx.cached_partial(val_step, training_state.model)

_, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)
train_ds = Dataset.create(**train_dataset)
val_ds = Dataset.create(**val_dataset)

key = jax.random.key(seed)

loss_sum = 0
for step in range(log_every*100):
    batch = train_ds.sample(batch_size)
    x_1 = batch["observations"].astype(jnp.float32) / 255.0

    key, key_update = jax.random.split(key)
    loss = train_step_cached(x_1, key)

    loss_sum += loss
    if step%log_every == (log_every-1):
        train_loss = loss_sum/log_every

        val_batch = val_ds.sample(batch_size)
        val_x = val_batch["observations"].astype(jnp.float32) / 255.0
        val_loss = val_step_cached(val_x, key)

        print(step, train_loss, val_loss)
        loss_sum = 0