from octo.model.octo_model import OctoModel
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
print(model.get_pretty_spec())

# download one example BridgeV2 image
IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
plt.imshow(img)

# create obs & task dict, run inference
import jax
# add batch + time horizon 1
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the fork"])
import ipdb; ipdb.set_trace()
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
print(action)   # [batch, action_chunk, action_dim]