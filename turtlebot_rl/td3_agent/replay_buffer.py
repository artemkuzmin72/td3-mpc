import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size circular replay buffer for off-policy RL.

    Stores (s, a, r, s', done) transitions as pre-allocated
    NumPy arrays for zero-copy batch sampling.

    Usage
    -----
        buf = ReplayBuffer(capacity=500_000)
        buf.init(state_dim=24, action_dim=2)
        buf.add(state, action, reward, next_state, done)
        batch = buf.sample(batch_size=256)
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self._ptr  = 0
        self._size = 0

        self.states      = None
        self.actions     = None
        self.rewards     = None
        self.next_states = None
        self.dones       = None

    # ------------------------------------------------------------------
    def init(self, state_dim: int, action_dim: int):
        cap = self.capacity
        self.states      = np.zeros((cap, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((cap, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((cap, 1),          dtype=np.float32)
        self.next_states = np.zeros((cap, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((cap, 1),          dtype=np.float32)

    # ------------------------------------------------------------------
    def add(self, state, action, reward, next_state, done):
        i = self._ptr
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = done

        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int,
               device: torch.device = torch.device("cpu")) -> dict:
        idx = np.random.randint(0, self._size, size=batch_size)

        def _t(arr):
            return torch.FloatTensor(arr[idx]).to(device)

        return {
            "state":      _t(self.states),
            "action":     _t(self.actions),
            "reward":     _t(self.rewards),
            "next_state": _t(self.next_states),
            "done":       _t(self.dones),
        }

    def save(self, path: str):
        import os
        path = os.path.expanduser(path)
        np.savez_compressed(
            path,
            states      = self.states     [:self._size],
            actions     = self.actions    [:self._size],
            rewards     = self.rewards    [:self._size],
            next_states = self.next_states[:self._size],
            dones       = self.dones      [:self._size],
            ptr         = np.array([self._ptr]),
            size        = np.array([self._size]),
        )
        print(f"[ReplayBuffer] Saved {self._size} transitions -> {path}.npz")

    # ------------------------------------------------------------------
    def load(self, path: str, keep_existing: bool = False):
        import os
        path = os.path.expanduser(path)
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path)
        loaded_size = int(data["size"][0])

        if self.states is None:
            raise RuntimeError("Вызови init() перед load()")

        if not keep_existing:
            # Полная замена — копируем загруженные данные с начала буфера
            n = min(loaded_size, self.capacity)
            self.states     [:n] = data["states"]     [:n]
            self.actions    [:n] = data["actions"]    [:n]
            self.rewards    [:n] = data["rewards"]    [:n]
            self.next_states[:n] = data["next_states"][:n]
            self.dones      [:n] = data["dones"]      [:n]
            self._size = n
            self._ptr  = n % self.capacity

        else:
            # Смешивание — добавляем загруженные переходы один за другим
            # (уважаем circular buffer логику)
            n = min(loaded_size, self.capacity)
            for i in range(n):
                self.add(
                    data["states"][i],
                    data["actions"][i],
                    data["rewards"][i],
                    data["next_states"][i],
                    float(data["dones"][i]),
                )

        print(f"[ReplayBuffer] Loaded {loaded_size} transitions from {path} "
              f"(keep_existing={keep_existing})  buffer size now: {self._size}")

    # ==================================================================
    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (f"ReplayBuffer(capacity={self.capacity}, "
                f"size={self._size}, ptr={self._ptr})")
