"""Head-to-head agent eval.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/bots.py --agent1 strategic --agent2 jidan --games 200
    PYTHONPATH=ml/src python ml/scripts/eval/bots.py --agent1 partner_pimc --agent2 jidan --games 100 --n-det 10
    PYTHONPATH=ml/src python ml/scripts/eval/bots.py --agent1 partner_oracle --agent2 jidan --games 200 --checkpoint ml/checkpoints/jidan_policy.pt --no-search
    PYTHONPATH=ml/src python ml/scripts/eval/bots.py --agent1 partner_oracle --agent2 jidan --games 200 --checkpoint ml/checkpoints/jidan_policy.pt --seat-rotate
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tqdm import tqdm

from guandan.agents import make_agent
from guandan.agents.partner_oracle_bot import PartnerOracleBot
from guandan.agents.partner_pimc_bot import PartnerPIMCBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv


def build_agent(name: str, n_det: int, n_cands: int, top_k: int, checkpoint: str | None,
                no_search: bool, no_belief: bool, use_value_leaf: bool = False,
                policy_checkpoint: str | None = None):
    if name == "partner_pimc":
        return PartnerPIMCBot(level_rank=Rank.TWO, n_det=n_det, n_cands=n_cands)
    if name == "partner_oracle":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for partner_oracle")
        import torch
        from guandan.azguan import QValueNet
        agent = PartnerOracleBot(
            checkpoint_path=checkpoint,
            level_rank=Rank.TWO,
            use_search=not no_search,
            n_det=n_det,
            top_k=top_k,
            use_belief=not no_belief,
            use_value_leaf=use_value_leaf,
        )
        if policy_checkpoint is not None:
            # Hybrid oracle: separate clean policy net for top-K candidate selection.
            # agent._search.value_net still holds the original V-head (bound at init).
            pol_ckpt = torch.load(policy_checkpoint, map_location=agent.device, weights_only=True)
            pol_cfg = pol_ckpt["config"]
            pol_net = QValueNet(
                d_state=pol_cfg["d_state"],
                d_action=pol_cfg["d_action"],
                hidden=pol_cfg["hidden"],
            ).to(agent.device)
            pol_net.load_state_dict(pol_ckpt["state_dict"], strict=False)
            pol_net.eval()
            agent.net = pol_net
        return agent
    return make_agent(name, level_rank=Rank.TWO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head agent eval")
    parser.add_argument("--agent1", required=True, help="Team {0,2} agent name")
    parser.add_argument("--agent2", required=True, help="Team {1,3} agent name")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--n-det", type=int, default=20,
                        help="PartnerPIMCBot/PartnerOracleBot: determinizations per move (default 20)")
    parser.add_argument("--n-cands", type=int, default=10,
                        help="PartnerPIMCBot: max candidates pre-filter (default 10)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for partner_oracle agent")
    parser.add_argument("--top-k", type=int, default=3,
                        help="partner_oracle: top-K candidates from policy to search (default 3)")
    parser.add_argument("--no-search", action="store_true",
                        help="partner_oracle: disable PIMC search, use pure policy argmax")
    parser.add_argument("--no-belief", action="store_true",
                        help="partner_oracle: disable BeliefModel in determinization (uniform sampling)")
    parser.add_argument("--use-value-leaf", action="store_true",
                        help="partner_oracle: use V(s) at leaf instead of Jidan rollouts (AZ gen-2+)")
    parser.add_argument("--policy-checkpoint", default=None,
                        help="partner_oracle: separate clean policy checkpoint for top-K candidate "
                             "selection (hybrid oracle; keeps value-leaf from --checkpoint)")
    parser.add_argument("--seat-rotate", action="store_true",
                        help="Alternate agent1 between seats {0,2} and {1,3} each game")
    args = parser.parse_args()

    a1 = build_agent(args.agent1, args.n_det, args.n_cands, args.top_k, args.checkpoint,
                     args.no_search, args.no_belief, args.use_value_leaf, args.policy_checkpoint)
    a2 = build_agent(args.agent2, args.n_det, args.n_cands, args.top_k, args.checkpoint,
                     args.no_search, args.no_belief, args.use_value_leaf)
    env = GuanDanEnv()
    wins = 0

    for i in tqdm(range(args.games), desc=f"{args.agent1} vs {args.agent2}"):
        env.reset()
        # With --seat-rotate, agent1 alternates seats to cancel positional bias.
        if args.seat_rotate and i % 2 == 1:
            a1_seats = {1, 3}
        else:
            a1_seats = {0, 2}

        while not env.done:
            p = env.current_player
            agent = a1 if p in a1_seats else a2
            env.step(agent.act(env, p))

        rewards = env.get_rewards()
        a1_seat = next(iter(a1_seats))
        if rewards[a1_seat] > 0:
            wins += 1

    wr = wins / args.games
    z = 1.96
    n = args.games
    center = (wr + z * z / (2 * n)) / (1 + z * z / n)
    margin = (z * math.sqrt(wr * (1 - wr) / n + z * z / (4 * n * n))) / (1 + z * z / n)

    print(f"\n{args.agent1} WR vs {args.agent2}: {wr:.1%}  "
          f"(95% CI [{max(0, center - margin):.1%}, {min(1, center + margin):.1%}])"
          f"  ({wins}/{args.games})")


if __name__ == "__main__":
    main()
