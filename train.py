import atexit
import pathlib
import sys
import warnings

import hydra
import numpy as np
import torch
from tensordict import TensorDict

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def prefill_from_demos(config, replay_buffer, agent):
    """Load demo episodes from .npz files into the replay buffer.

    Successful demos (length <= time_limit) are loaded first. Demo episode IDs
    start at 10_000 to avoid collision with online training IDs (0..env_num-1).
    """
    demodir = pathlib.Path(config.env.demodir).expanduser()
    time_limit = int(config.env.time_limit)
    maxnumdemos = int(config.env.maxnumdemos)

    paths = sorted(demodir.glob("*.npz"))
    if not paths:
        print(f"Warning: no .npz demos found in {demodir}")
        return

    # Load all episodes, split successful (short) first.
    successful, failed = [], []
    for p in paths:
        try:
            ep = dict(np.load(p, allow_pickle=False))
        except Exception as e:
            print(f"Could not load demo {p}: {e}")
            continue
        (successful if len(ep["reward"]) <= time_limit else failed).append(ep)

    episodes = (successful + failed)[:maxnumdemos]
    print(f"Demo prefill: {len(successful)} successful + {len(failed)} failed demos found, loading {len(episodes)}")

    # Determine latent shapes from the agent.
    init_state = agent.get_initial_state(1)
    stoch_shape = init_state["stoch"].shape[1:]  # (S, K)
    deter_shape = init_state["deter"].shape[1:]  # (D,)

    # Demo episode IDs start at 10_000 to avoid collision with online IDs (0..env_num-1).
    demo_ep_id_start = 10_000
    n_steps = 0

    for ep_idx, ep in enumerate(episodes):
        ep_id = demo_ep_id_start + ep_idx
        T = len(ep["reward"])

        for t in range(T):
            trans = TensorDict(
                {
                    "image": torch.from_numpy(ep["image"][t]).unsqueeze(0),
                    "state": torch.from_numpy(ep["state"][t]).float().unsqueeze(0),
                    "is_first": torch.tensor([ep["is_first"][t]], dtype=torch.bool),
                    "is_last": torch.tensor([ep["is_last"][t]], dtype=torch.bool),
                    "is_terminal": torch.tensor([ep["is_terminal"][t]], dtype=torch.bool),
                    "reward": torch.tensor([ep["reward"][t]], dtype=torch.float32),
                    "action": torch.from_numpy(ep["action"][t]).float().unsqueeze(0),
                    "episode": torch.tensor([ep_id], dtype=torch.int32),
                    "stoch": torch.zeros(1, *stoch_shape),
                    "deter": torch.zeros(1, *deter_shape),
                },
                batch_size=[1],
            )
            replay_buffer.add_transition(trans)
            n_steps += 1

    print(f"Demo prefill complete: loaded {len(episodes)} demos ({n_steps} steps) from {demodir}")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir, config=config)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    if getattr(config.env, "demodir", None):
        prefill_from_demos(config, replay_buffer, agent)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
