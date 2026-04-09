import numpy as np
import gymnasium as gym

SUCCESS_REWARD_SCALAR_VALUE = 100.0


class ManiSkill(gym.Env):
    """ManiSkill environment wrapper for r2dreamer.

    Wraps ManiSkill's PickCube-v1 with dual-camera RGB (base + hand = 6ch),
    7-dim tcp_pose state, and custom success detection.
    """

    metadata = {}

    def __init__(
        self,
        task="PickCube-v1",
        size=(128, 128),
        action_repeat=1,
        seed=None,
        time_limit=350,
        obs_mode="rgb+state_dict",
        control_mode="pd_ee_delta_pos",
    ):
        assert task in ("PickCube-v1",), f"Unsupported ManiSkill task: {task}"
        import mani_skill
        from mani_skill.utils.wrappers.action_repeat import ActionRepeatWrapper

        self.size = tuple(size)
        self.seed = seed

        # max_episode_steps must be set explicitly; None makes ManiSkill use
        # its own short default (50 for PickCube), bypassing our TimeLimit wrapper.
        self._env = gym.make(
            task,
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            max_episode_steps=time_limit,
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            robot_uids="panda_wristcam",
        )

        if action_repeat > 1:
            self._env = ActionRepeatWrapper(self._env, repeat=action_repeat)

        self.obs_mode = obs_mode
        self._img_channels = 6  # base_camera RGB (3) + hand_camera RGB (3)
        self._state_dim = 7  # tcp_pose: position (3) + quaternion (4)

    # ------------------------------------------------------------------
    # Observation and action spaces
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        h, w = self.size
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (h, w, self._img_channels), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (self._state_dim,), dtype=np.float32),
            "log_success": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        })

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env

    # ------------------------------------------------------------------
    # Observation extraction helpers
    # ------------------------------------------------------------------

    def _extract_image(self, obs):
        """Return base_camera RGB + hand_camera RGB as (H, W, 6) uint8."""
        import torch
        import torch.nn.functional as F

        sd = obs["sensor_data"]
        base_rgb = sd["base_camera"]["rgb"][0].float()  # (H, W, 3)
        hand_rgb = sd["hand_camera"]["rgb"][0].float()  # (H, W, 3)

        combined = torch.cat([base_rgb, hand_rgb], dim=-1)  # (H, W, 6)
        if combined.shape[0] != self.size[0] or combined.shape[1] != self.size[1]:
            t = combined.permute(2, 0, 1).unsqueeze(0)  # (1, 6, H, W)
            t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)
            combined = t.squeeze(0).permute(1, 2, 0)  # (H, W, 6)

        return combined.clamp(0, 255).byte().cpu().numpy()

    def _extract_reduced_state(self, obs):
        """Return tcp_pose as (7,) float32."""
        import torch

        tcp = obs["extra"]["tcp_pose"][0]
        if torch.is_tensor(tcp):
            return tcp.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.asarray(tcp).reshape(-1).astype(np.float32)

    def custom_success_check(self, obs):
        """Looser success criterion: within goal threshold, grasped."""
        import numpy as np

        distance = np.linalg.norm(
            obs["extra"]["obj_to_goal_pos"].detach().cpu().numpy()[0]
        )
        return bool(obs["extra"]["is_grasped"]) and distance < self._env.unwrapped.goal_thresh

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", self.seed)
        obs, _ = self._env.reset(seed=seed, **kwargs)

        return {
            "image": self._extract_image(obs),
            "state": self._extract_reduced_state(obs),
            "log_success": np.array([0.0], dtype=np.float32),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "reward": np.float32(0.0),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        reward = np.float32(reward)[0]

        success = bool(info.get("success", False))
        if self.custom_success_check(obs):
            success = True
            truncated = True

        if success:
            reward = np.float32(SUCCESS_REWARD_SCALAR_VALUE)

        done = terminated or truncated

        out_obs = {
            "image": self._extract_image(obs),
            "state": self._extract_reduced_state(obs),
            "log_success": np.array([1.0 if success else 0.0], dtype=np.float32),
            # is_terminal = True only on task success (not time-limit truncation)
            # This preserves correct value-bootstrapping semantics in DreamerV3.
            "is_first": False,
            "is_last": bool(done),
            "is_terminal": bool(terminated or success),
            "reward": reward,
        }
        return out_obs, reward, bool(done), info

    def render(self):
        return self._env.render()
