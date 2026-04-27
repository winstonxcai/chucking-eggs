# Methodology

## 1. Problem Setting: Partner-Visible GuanDan

We model GuanDan as a four-player, two-team, imperfect-information card game. Players sit in fixed partnerships, with teammates opposite each other. The game uses two standard decks plus four jokers, for 108 cards total, and each hand is governed by a current level/rank card whose heart-suit instances act as wild cards. GuanDan proceeds as a climbing/shedding game in which players play legal combinations, pass, contest tricks, and try to help their partnership finish before the opposing partnership. ([Pagat][1])

Let the players be:

[
\mathcal{N} = {0,1,2,3}
]

with teams:

[
T_A = {0,2}, \qquad T_B = {1,3}.
]

For each player (i), define the partner function:

[
p(i) = (i + 2) \bmod 4.
]

The proposed setting is **partner-visible GuanDan**. At decision time, the acting player observes:

[
\text{own hand} + \text{partner hand} + \text{public game history},
]

but does **not** observe the opponents’ private hands.

This produces a middle ground between standard GuanDan and full perfect-information GuanDan:

| Setting                                | Own hand | Partner hand | Opponent hands |
| -------------------------------------- | -------: | -----------: | -------------: |
| Standard imperfect-information GuanDan |  visible |       hidden |         hidden |
| **Partner-visible GuanDan**            |  visible |      visible |         hidden |
| Full perfect-information GuanDan       |  visible |      visible |        visible |

The deployed policy is only allowed to use the partner-visible observation. Opponent hands may be used by a privileged critic during training, but never by the deployed actor.

---

## 2. Relation to Perfect-Training / Imperfect-Execution

The method follows the key PerfectDou design principle: the policy network must receive only information available at execution time, while the value network may receive privileged information during training. In PerfectDou, this is described as a perfect-training-imperfect-execution framework where the policy uses imperfect information but the value network can use additional perfect information, such as hidden cards. The original system trains this actor-critic setup with PPO and GAE. ([arXiv][2])

For GuanDan, we adapt this into:

> **Perfect-training / partner-visible-execution.**

The actor receives only partner-visible information:

[
\pi_\theta(a_t \mid o^{PV}_{i,t}),
]

while the critic may receive the full state during training:

[
V_\psi(s_t, i).
]

The deployed model discards the critic and uses only:

[
\pi_\theta(a_t \mid \text{own hand}, \text{partner hand}, \text{public history}).
]

This preserves the user-preferred information structure: teammates know partner cards, but not opponent cards.

---

## 3. State, Observation, and Information Structure

Let the full environment state at time (t) be:

[
s_t =
\left(
H^t_0, H^t_1, H^t_2, H^t_3,
P_t,
C_t,
L_t,
F_t
\right),
]

where:

* (H^t_i): remaining hand of player (i)
* (P_t): public play history
* (C_t): current trick context
* (L_t): current level/rank-card information
* (F_t): finish-order and player-status information

For acting player (i), define the **partner-visible observation**:

[
o^{PV}_{i,t}
============

# \phi^{PV}_i(s_t)

\left(
H^t_i,
H^t_{p(i)},
P_t,
C_t,
L_t,
F_t,
n^t_0,n^t_1,n^t_2,n^t_3,
\mathcal{A}(s_t)
\right),
]

where (n^t_j = |H^t_j|) is the remaining card count for player (j), and (\mathcal{A}(s_t)) is the set of legal actions available to the acting player.

The observation explicitly excludes:

[
H^t_j \quad \text{for all } j \notin {i,p(i)}.
]

Thus, the actor’s information set is refined relative to standard GuanDan because partner cards are visible, but it remains imperfect because opponent cards are hidden.

---

## 4. Environment and Rule Engine

The environment must implement a rule-complete GuanDan simulator. This component is treated as part of the methodology rather than a minor engineering detail, because invalid legal-action generation would corrupt the learning signal.

The environment must support:

* four-player fixed-partnership play;
* two-deck card representation;
* level-card ranking;
* heart-level-card wildcards;
* tribute / return-tribute rules, if included in the chosen rule variant;
* trick leadership and pass logic;
* finish-order tracking;
* rank advancement after each hand;
* legal combination generation;
* action comparison and bomb hierarchy.

The legal action generator must enumerate all legal actions available from the acting player’s actual hand. GuanDan ordinary combinations include singles, pairs, triples, full houses, straights, tubes, and plates; bombs include multi-card rank bombs, straight flushes, and the four-joker bomb. Wild cards must be represented by both their physical cards and their declared semantic role in the played combination. ([Pagat][1])

Each action (a) should therefore store:

[
a = (\text{physical cards}, \text{declared combination type}, \text{comparison key}).
]

This avoids ambiguity when the same physical cards can be interpreted as different legal combinations due to wild cards.

---

## 5. Action Space Design

Because GuanDan has a large and variable legal-action space, the policy should not use a fixed flat action classifier over all possible card combinations.

Instead, the model uses **candidate-action scoring**.

At each decision point, the environment enumerates:

[
\mathcal{A}_t = {a_1,a_2,\dots,a_m}.
]

The policy scores only these legal actions:

[
z_k = f_\theta(o^{PV}_{i,t}, a_k),
]

and samples from the masked softmax:

[
\pi_\theta(a_k \mid o^{PV}_{i,t})
=================================

\frac{\exp(z_k)}
{\sum_{a_j \in \mathcal{A}_t} \exp(z_j)}.
]

This formulation has three advantages:

1. illegal actions are never proposed;
2. the model naturally handles variable action counts;
3. action representations can include rich structural information such as combination type, length, rank, bomb class, and wildcard usage.

---

## 6. Representation Design

### 6.1 Card-Zone Encoding

Cards are encoded by zone. Recommended zones are:

[
Z =
{
\text{own hand},
\text{partner hand},
\text{played cards},
\text{current trick},
\text{last non-pass action}
}.
]

Each zone is represented as either:

[
54\text{-type count vector}
]

or

[
108\text{-instance binary vector}.
]

The 54-type count representation is usually sufficient, but the 108-instance representation may be useful if duplicate physical cards need to be tracked separately.

Each card encoding includes:

* rank;
* suit;
* joker indicator;
* current-level-card indicator;
* current wild-card indicator;
* natural rank;
* level-adjusted rank.

### 6.2 Public State Encoding

The public state vector includes:

[
g_t =
\left[
\text{acting seat},
\text{relative seat positions},
\text{team identity},
\text{current level},
\text{current leader},
\text{pass sequence},
\text{remaining hand sizes},
\text{finish order},
\text{tribute state},
\text{current trick type},
\text{current trick rank}
\right].
]

Player identities should be encoded relatively:

[
\text{self}, \text{partner}, \text{left opponent}, \text{right opponent}
]

rather than as absolute seat IDs. This helps a single shared policy generalize across all seats.

### 6.3 Action Encoding

Each candidate action (a_k) is encoded as:

[
e(a_k) =
[
\text{card-count vector},
\text{combination type},
\text{primary rank},
\text{kicker ranks},
\text{sequence length},
\text{bomb type},
\text{wildcard usage},
\text{is-pass flag}
].
]

For actions involving wild cards, the declared substituted value is encoded separately from the physical card identity.

---

## 7. Policy and Value Architecture

The actor uses a state encoder, an action encoder, and a candidate-action scoring head.

[
h_s = E_s(o^{PV}_{i,t})
]

[
h_a = E_a(a_k)
]

[
z_k =
\operatorname{MLP}
\left(
[
h_s,
h_a,
h_s \odot h_a,
|h_s-h_a|
]
\right).
]

The policy is:

[
\pi_\theta(a_k \mid o^{PV}_{i,t})
=================================

\operatorname{softmax}_{a_k \in \mathcal{A}_t}(z_k).
]

The critic has two possible variants.

### Variant A: Strict Partner-Visible Critic

The critic receives the same information as the actor:

[
V_\psi(o^{PV}_{i,t}).
]

This variant tests the value of partner visibility alone.

### Variant B: Privileged Full-State Critic

The critic receives the full state:

[
V_\psi(s_t, i),
]

including opponent hands during training.

The actor still receives only:

[
o^{PV}_{i,t}.
]

This is the direct PerfectDou-style adaptation. The privileged critic improves advantage estimation, while the actor remains deployable in the partner-visible game.

The main proposed method is Variant B, with Variant A used as an ablation.

---

## 8. Reward Design

The primary reward should be based on **team outcome**, not individual finishing position alone.

Let (\Delta L_T) denote the rank advancement earned by team (T) after a hand under the selected GuanDan rule variant.

For player (i), define:

[
T(i) = \text{team of player } i,
]

[
\bar{T}(i) = \text{opposing team}.
]

The terminal hand reward is:

[
R_i
===

\frac{
\Delta L_{T(i)} - \Delta L_{\bar{T}(i)}
}{
R_{\max}
}.
]

Here, (R_{\max}) normalizes the reward into a stable range such as ([-1,1]).

This reward encourages:

* winning the hand;
* helping the partner finish early;
* maximizing rank advancement, not merely being first individually.

If the rule variant distinguishes between different finish patterns, the simulator should expose the advancement table as a configurable rule. For example:

[
1\text{-}2 \text{ finish} > 1\text{-}3 \text{ finish} > 1\text{-}4 \text{ finish}.
]

The exact advancement values should be taken from the chosen competition rule set.

### Optional Potential-Based Shaping

A shaped reward may be added during training:

[
r'_t
====

r_t
+
\eta
\left(
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right),
]

where (\Phi(s_t)) is a heuristic team potential, such as negative estimated minimum remaining plays for the team.

A possible potential is:

[
\Phi_i(s_t)
===========

*

D_{T(i)}(s_t)
+
D_{\bar{T}(i)}(s_t),
]

where (D_T(s_t)) estimates the minimum number of turns required for team (T) to shed its remaining cards.

This shaping term should be treated as an ablation, not as the default, because poor shaping can create brittle behaviors.

---

## 9. Training Objective

Training uses multi-agent self-play with PPO and GAE.

For each transition, store:

[
(o^{PV}*{i,t}, s_t, a_t, r_t, o^{PV}*{i,t+1}, s_{t+1}, \log \pi_{\theta_{\text{old}}}(a_t \mid o^{PV}_{i,t})).
]

The privileged state (s_t) is stored only for critic training.

The PPO probability ratio is:

[
\rho_t(\theta)
==============

\frac{
\pi_\theta(a_t \mid o^{PV}*{i,t})
}{
\pi*{\theta_{\text{old}}}(a_t \mid o^{PV}_{i,t})
}.
]

The clipped actor objective is:

[
\mathcal{L}_{\pi}
=================

\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat{A}_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
]

The critic loss is:

[
\mathcal{L}_V
=============

\mathbb{E}*t
\left[
\left(
V*\psi(x_t) - \hat{R}_t
\right)^2
\right],
]

where:

[
x_t =
\begin{cases}
o^{PV}_{i,t}, & \text{strict partner-visible critic},\
s_t, & \text{privileged full-state critic}.
\end{cases}
]

The total loss is:

[
\mathcal{L}
===========

*

\mathcal{L}_{\pi}
+
c_V \mathcal{L}_V
-----------------

c_H \mathcal{H}(\pi_\theta),
]

where (\mathcal{H}) is an entropy bonus over legal actions.

---

## 10. Advantage Estimation

Advantages are computed using GAE:

[
\delta_t
========

r_t
+
\gamma V_\psi(x_{t+1})
----------------------

V_\psi(x_t),
]

[
\hat{A}_t
=========

\sum_{l=0}^{\infty}
(\gamma \lambda)^l
\delta_{t+l}.
]

When using the privileged critic, the value network can condition on opponent cards during training:

[
x_t = s_t.
]

This allows the critic to estimate lower-variance advantages from the full hidden state. However, the policy update still optimizes:

[
\pi_\theta(a_t \mid o^{PV}_{i,t}),
]

so no opponent-card information is available at execution.

---

## 11. Optional Opponent-Belief Auxiliary Task

To improve hidden-state reasoning, the actor encoder may include an auxiliary prediction head trained to infer opponent card distributions from partner-visible observations.

Let the two opponents of player (i) be (j) and (k). The auxiliary head predicts:

[
\hat{H}_j, \hat{H}_k
====================

B_\omega(h_s).
]

The target is the true opponent hands from the simulator:

[
H_j, H_k.
]

The auxiliary loss is:

[
\mathcal{L}_{belief}
====================

\operatorname{BCE}(\hat{H}_j, H_j)
+
\operatorname{BCE}(\hat{H}_k, H_k).
]

The combined objective becomes:

[
\mathcal{L}
===========

*

\mathcal{L}_{\pi}
+
c_V \mathcal{L}*V
+
c_B \mathcal{L}*{belief}
------------------------

c_H \mathcal{H}(\pi_\theta).
]

At inference, the belief head may be discarded or retained internally, but the actor still receives only partner-visible observations.

---

## 12. Training Pipeline

### Stage 1: Rule-Complete Simulator

Implement the GuanDan environment with exact legal-action generation, action comparison, wildcard handling, trick resolution, finish-order tracking, and rank advancement.

The simulator outputs:

[
s_t,\quad o^{PV}_{i,t},\quad \mathcal{A}_t,\quad r_t.
]

### Stage 2: Standard Imperfect-Information Baseline

Train a baseline actor that observes only:

[
o^{STD}_{i,t}
=============

(\text{own hand}, \text{public history}).
]

This baseline quantifies the difficulty of standard hidden-partner GuanDan.

### Stage 3: Partner-Visible Actor-Critic

Train the strict partner-visible model:

[
\pi_\theta(a \mid o^{PV}*{i,t}), \qquad V*\psi(o^{PV}_{i,t}).
]

This isolates the value of revealing partner cards.

### Stage 4: Partner-Visible PTIE

Train the main model:

[
\pi_\theta(a \mid o^{PV}*{i,t}), \qquad V*\psi(s_t,i).
]

The actor sees only partner-visible information, while the critic sees full game state during training.

### Stage 5: Fine-Tuning Without Privileged Critic

Optionally fine-tune the actor using a partner-visible critic:

[
V_\psi(o^{PV}_{i,t})
]

after the main PTIE training stage.

This reduces dependency on a critic that had access to opponent cards and can make the final policy more robust under partner-visible execution.

---

## 13. Evaluation Protocol

Evaluation should use fixed held-out deals and seat rotations.

For each matchup:

1. sample a held-out deck seed;
2. play the same deal under multiple seat assignments;
3. rotate team positions;
4. report confidence intervals over many hands.

The primary metrics are:

[
\text{Average rank advancement per hand}
]

[
\text{Match win rate}
]

[
\text{1-2 / 1-3 / 1-4 finish distribution}
]

[
\text{Average promotion differential}
]

[
\text{Bomb efficiency}
]

[
\text{Partner-assisted finish rate}.
]

A partner-assisted finish can be operationalized as a sequence where player (i):

* gains control;
* leads a combination favorable to partner (p(i));
* partner subsequently sheds or finishes.

---

## 14. Ablation Study

The core ablations are:

| Model            | Actor observation                        | Critic observation | Purpose                                  |
| ---------------- | ---------------------------------------- | ------------------ | ---------------------------------------- |
| STD-AC           | own hand + public history                | same               | standard imperfect-information baseline  |
| PV-AC            | own hand + partner hand + public history | same               | effect of partner visibility             |
| PV-PTIE          | own hand + partner hand + public history | full state         | PerfectDou-style privileged critic       |
| PV-PTIE + belief | own hand + partner hand + public history | full state         | effect of opponent-belief auxiliary loss |
| FI upper bound   | full state                               | full state         | non-deployable performance ceiling       |

The main comparison is:

[
\text{PV-AC} \quad \text{vs.} \quad \text{STD-AC}.
]

This measures the value of partner-card visibility.

The second comparison is:

[
\text{PV-PTIE} \quad \text{vs.} \quad \text{PV-AC}.
]

This measures whether privileged full-state critic training improves the partner-visible policy.

The full-information model is not a valid deployed agent. It is used only as an upper bound.

---

## 15. Deployment Policy

The deployed agent is:

[
a_t
\sim
\pi_\theta
\left(
a_t
\mid
H_i^t,
H_{p(i)}^t,
P_t,
C_t,
L_t,
F_t,
\mathcal{A}(s_t)
\right).
]

At deployment, the model does not receive:

[
H_j^t
\quad
\text{for opponents } j \notin {i,p(i)}.
]

The critic, opponent-belief targets, and any full-state training features are removed or disabled during execution.

Thus, the final agent plays the **partner-visible GuanDan variant**, not full perfect-information GuanDan.

[1]: https://www.pagat.com/climbing/guan_dan.html "Guan Dan - card game rules"
[2]: https://arxiv.org/html/2203.16406v7 "PerfectDou: Dominating DouDizhu with Perfect Information Distillation"
