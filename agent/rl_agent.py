import numpy as np
from config import (
    RL_HIDDEN_DIM, RL_GAMMA, RL_GAE_LAMBDA, RL_CLIP_EPS,
    RL_ENT_COEF, RL_VF_COEF, RL_MAX_GRAD_NORM, RL_EPOCHS,
    RL_BATCH_SIZE, RL_LEARNING_RATE, JOINT_LIMITS,
)


def layer_init(shape, scale=np.sqrt(2)):
    return np.random.randn(*shape).astype(np.float32) * scale * 0.01


class RollingBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.logprobs = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def store(self, obs, action, reward, done, value, logprob):
        i = self.idx % self.capacity
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.logprobs[i] = logprob
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True

    def compute_gae(self, last_value):
        n = self.capacity if self.full else self.idx
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_nonterminal = 1.0
            else:
                next_val = self.values[(t + 1) % self.capacity]
                next_nonterminal = 1.0 - self.dones[t]
            delta = (self.rewards[t] + RL_GAMMA * next_val * next_nonterminal
                     - self.values[t])
            gae = delta + RL_GAMMA * RL_GAE_LAMBDA * next_nonterminal * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_batches(self, batch_size):
        n = self.capacity if self.full else self.idx
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        inds = np.arange(n)
        np.random.shuffle(inds)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mb = inds[start:end]
            yield (
                self.obs[mb], self.actions[mb],
                self.logprobs[mb], adv[mb], self.returns[mb]
            )


class PPOAgent:
    def __init__(self, obs_dim=13, act_dim=6):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        h = RL_HIDDEN_DIM
        self.pi_w1 = layer_init((obs_dim, h))
        self.pi_b1 = np.zeros(h, dtype=np.float32)
        self.pi_w2 = layer_init((h, h))
        self.pi_b2 = np.zeros(h, dtype=np.float32)
        self.pi_w3 = layer_init((h, act_dim), scale=0.01)
        self.pi_b3 = np.zeros(act_dim, dtype=np.float32)
        self.pi_logstd = np.zeros(act_dim, dtype=np.float32)

        self.v_w1 = layer_init((obs_dim, h))
        self.v_b1 = np.zeros(h, dtype=np.float32)
        self.v_w2 = layer_init((h, h))
        self.v_b2 = np.zeros(h, dtype=np.float32)
        self.v_w3 = layer_init((h, 1), scale=0.01)
        self.v_b3 = np.zeros(1, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def _pi(self, obs):
        h = self._relu(obs @ self.pi_w1 + self.pi_b1)
        h = self._relu(h @ self.pi_w2 + self.pi_b2)
        mean = h @ self.pi_w3 + self.pi_b3
        return mean, self.pi_logstd

    def _v(self, obs):
        h = self._relu(obs @ self.v_w1 + self.v_b1)
        h = self._relu(h @ self.v_w2 + self.v_b2)
        return (h @ self.v_w3 + self.v_b3).flatten()

    def act(self, obs, greedy=False):
        mean, logstd = self._pi(obs)
        if greedy:
            return mean.copy()
        std = np.exp(logstd)
        return (mean + std * np.random.randn(self.act_dim)).astype(np.float32)

    def act_with_metrics(self, obs):
        mean, logstd = self._pi(obs)
        std = np.exp(logstd)
        action = (mean + std * np.random.randn(self.act_dim)).astype(np.float32)
        value = self._v(obs)[0]
        logprob = self._gaussian_logprob(action, mean, std)
        return action, value, float(logprob)

    def _gaussian_logprob(self, action, mean, std):
        return float(np.sum(
            -0.5 * ((action - mean) / (std + 1e-8)) ** 2
            - 0.5 * np.log(2 * np.pi)
            - np.log(std + 1e-8)
        ))

    def update(self, buffer, last_value):
        buffer.compute_gae(last_value)
        n = buffer.capacity if buffer.full else buffer.idx
        pi_opt = AdamOptimizer(RL_LEARNING_RATE)
        v_opt = AdamOptimizer(RL_LEARNING_RATE)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        batches = 0

        for _ in range(RL_EPOCHS):
            for obs_b, act_b, old_logprob_b, adv_b, ret_b in buffer.get_batches(
                RL_BATCH_SIZE
            ):
                mean, logstd = self._pi(obs_b)
                std = np.exp(logstd)
                new_logprob = np.array([
                    self._gaussian_logprob(act_b[i], mean[i], std[i])
                    for i in range(len(act_b))
                ])
                ratio = np.exp(new_logprob - old_logprob_b)
                surr1 = ratio * adv_b
                surr2 = np.clip(ratio, 1 - RL_CLIP_EPS, 1 + RL_CLIP_EPS) * adv_b
                pi_loss = -np.mean(np.minimum(surr1, surr2))
                ent = np.mean(np.sum(
                    logstd + 0.5 * np.log(2 * np.pi * np.e), axis=1
                ))
                pi_loss -= RL_ENT_COEF * ent

                v_pred = self._v(obs_b)
                v_loss = np.mean((ret_b - v_pred) ** 2)

                total_pi_loss += float(pi_loss)
                total_v_loss += float(v_loss)
                batches += 1

                pi_opt.step(
                    lambda lr: self._sgd_step_pi(obs_b, act_b, adv_b, old_logprob_b,
                                                  lr, pi_loss, ret_b)
                )
                v_opt.step(
                    lambda lr: self._sgd_step_v(obs_b, ret_b, lr, v_loss)
                )

        return batches

    def _sgd_step_pi(self, obs_b, act_b, adv_b, old_logprob_b, lr, loss, ret_b):
        pass

    def get_params(self):
        params = {}
        for name in ["pi_w1", "pi_b1", "pi_w2", "pi_b2", "pi_w3", "pi_b3",
                      "pi_logstd", "v_w1", "v_b1", "v_w2", "v_b2", "v_w3", "v_b3"]:
            params[name] = getattr(self, name).copy()
        return params

    def set_params(self, params):
        for name, val in params.items():
            setattr(self, name, val)


class AdamOptimizer:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, closure):
        pass


def build_obs(env):
    ee = env.get_ee_pos()
    angles = env.get_joint_angles()
    ball = env.get_primary_ball_pos()
    obs = np.concatenate([ee, angles, ball]).astype(np.float32)
    return obs


def compute_reward(prev_dist, curr_dist, collision=False, grabbed=False):
    reward = 0.0
    if grabbed:
        reward += 50.0
    else:
        reward += (prev_dist - curr_dist) * 20.0
    if curr_dist < 0.05:
        reward += 200.0
    if collision:
        reward -= 10.0
    reward -= 0.1
    return reward
