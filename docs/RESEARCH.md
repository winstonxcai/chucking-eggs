# Why DART

DART exists to test whether a compact Deep Monte Carlo agent can learn
cooperative Guan Dan play when the state representation makes partnership
coordination explicit.

## Problem Setting

Guan Dan is a four-player, two-team card game with a 108-card double deck,
changing wild cards, bombs, and nontrivial incentives around helping a partner
finish. Absolute seat identity is less important than a player's role in the
current trick: leading, first responder, across from the leader, or last
responder.

## Approach

DART uses a shared Q-network with four trick-position heads. Every candidate
move is encoded relative to the current actor and routed through the head for
that actor's current trick role. This keeps the parameterization compact while
still letting the learner specialize value estimates by tactical position.

The training loop follows the DouZero / GuanZero family: many CPU actors
generate Monte Carlo samples, a learner trains on replay, and actors refresh
published weights asynchronously. DART differs by using partner-visible
features, role-normalized encoding, intra-actor lane batching, and a single
role-aware network instead of four absolute-seat networks.

## Design Tradeoffs

Partner visibility removes teammate-card inference from the problem so the
model can spend capacity on coordination. This makes the experiment cleaner as
a study of cooperative play, but it also means the result is not a strict
hidden-information deployment policy.

Pure Monte Carlo targets avoid target-network and bootstrapping complexity.
They are simple to reproduce and match the card-game lineage this project
builds on, but they require substantial actor throughput and can learn more
slowly than bootstrapped methods.

Role-aware shared heads reduce duplicated parameters and make actor inference
cheaper than maintaining separate full networks per seat. The tradeoff is that
all seats share most capacity, so bugs in role encoding affect the entire
policy rather than one isolated head.
