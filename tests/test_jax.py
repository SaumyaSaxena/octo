import jax
import jax.numpy as jnp
import flax.linen as nn

class AutoEncoder(nn.Module):
  def setup(self):
    self.encoder = nn.Dense(3)
    self.decoder = nn.Dense(5)

  def __call__(self, x):
    return self.decoder(self.encoder(x))

x1 = jnp.ones((16, 9))
ae = AutoEncoder()
variables1 = ae.init(jax.random.key(0), x1)
model = ae.bind(variables1)
z = model.encoder(x1)
x_reconstructed = model.decoder(z)

x2 = jnp.ones((16, 9))
ae = AutoEncoder()
variables2 = ae.init(jax.random.key(0), x2)
# variables_en = ae.init(jax.random.key(0), x, method=ae.encode)

encoded = model.apply(variables2, x2)