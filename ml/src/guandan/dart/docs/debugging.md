# Dart Debugging

For stepping through one training episode without the multiprocess trainer,
construct the model, encoder, replay buffer, and learner directly:

```python
import numpy as np

from guandan.dart.runtime.actor import all_latest_seats, play_episode
from guandan.dart.config import MODEL_TYPE_DART, TrainConfig, dart_qnet_config
from guandan.dart.data.buffer import RoleAwareReplayBuffer
from guandan.dart.model.encoding.role_encoder import (
    ROLE_ENCODE_CHANNEL_KEYS,
    RoleAwareStateActionEncoder,
)
from guandan.dart.model.q_network import DartQNet
from guandan.dart.runtime.learners.dart import DartLearner

cfg = TrainConfig(model_type=MODEL_TYPE_DART, batch_size=32, buffer_capacity=1_000)
q_net = DartQNet(dart_qnet_config(cfg))
encoder = RoleAwareStateActionEncoder(is_partner_visible=cfg.qnet.is_partner_visible)

samples = play_episode(
    q_nets=q_net,
    encoder=encoder,
    seats=all_latest_seats(0.1),
    seed=0,
    device="cpu",
)

buffer = RoleAwareReplayBuffer(capacity=1_000)
buffer.push_stacked(
    {k: np.stack([s.encoded[k] for s in samples], axis=0) for k in ROLE_ENCODE_CHANNEL_KEYS},
    np.asarray([s.mc_return for s in samples], dtype=np.float32),
)

learner = DartLearner(q_net=q_net, lr=1e-4, device="cpu")
metrics = learner.update(buffer, batch_size=32)
print(metrics)
```

This path exercises the production DART encoder, rollout, replay buffer, and
learner update while keeping all state in one process. For comparison-baseline
debugging, the GuanZero path remains available through `GuanZeroQNet`,
`StateActionEncoder`, `ReplayBuffer`, and `SeatLearner`.
