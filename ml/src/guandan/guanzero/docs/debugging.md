# GuanZero Debugging

For stepping through one training episode without the multiprocess trainer,
construct the model, encoder, replay buffer, and learner directly:

```python
from guandan.guanzero.runtime.actor import play_episode
from guandan.guanzero.data.buffer import ReplayBuffer
from guandan.guanzero.config import QNetConfig
from guandan.guanzero.model.encoder import StateActionEncoder
from guandan.guanzero.runtime.learner import Learner
from guandan.guanzero.model.q_network import init_seat_nets

q_nets = init_seat_nets(QNetConfig(hidden_lstm=64, hidden_mlp=128, n_mlp_layers=3))
encoder = StateActionEncoder()

samples = play_episode(
    q_nets=q_nets,
    encoder=encoder,
    epsilon=0.1,
    seed=0,
    device="cpu",
)

buffer = ReplayBuffer(capacity_per_player=1_000)
buffer.push(samples)

learner = Learner(q_nets=q_nets, lr=1e-4, device="cpu")
losses = learner.update(buffer, batch_size=8)
print(losses)
```

This path exercises the same encoder, rollout, replay buffer, and learner update
used by the distributed trainer, while keeping all state in one process.
