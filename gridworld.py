import numpy as np


class GridWorld:
    """
    Generalized M×N GridWorld environment.

    State space : S = {0, 1, ..., nrows*ncols - 1}
        Row-major indexing: state s  →  row = s // ncols,  col = s % ncols

    Actions     : 0=up, 1=down, 2=left, 3=right
    Dynamics    : deterministic (default) or stochastic.
                  Stochastic: P(intended) = 1 - 2*slip_prob,
                              P(each perpendicular direction) = slip_prob
    Reward      : -1 on every non-terminal transition (default),
                  or provided via reward_fn(state, action, next_state) -> float
    Policy      : initialised to the equiprobable random policy (0.25 each action)

    Parameters
    ----------
    N                : number of rows (grid side length when square)
    terminal_states  : set of in-grid terminal state indices, e.g. {0, N*N-1}
    ncols            : number of columns; defaults to N (square grid)
    stochastic       : if True, actions have stochastic outcomes
    slip_prob        : probability of slipping to each perpendicular direction
                       (used when stochastic=True; default 0.1 gives 0.8/0.1/0.1)
    reward_fn        : callable (state, action, next_state) -> float;
                       overrides the default -1 reward when provided
    boundary_terminal: dict {action_idx: reward} — when the agent tries to move
                       out of bounds in a listed direction, the episode ends with
                       the given reward instead of bouncing back.
                       Example for Q2.1: {0: +1.0, 1: -1.0}
    """

    ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
    _DELTAS  = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # (Δrow, Δcol)
    # Perpendicular (orthogonal) action pairs used for stochastic slip
    _PERP    = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

    def __init__(
        self,
        N: int,
        terminal_states: set,
        ncols: int = None,
        stochastic: bool = False,
        slip_prob: float = 0.1,
        reward_fn=None,
        boundary_terminal: dict = None,
    ):
        if N < 1:
            raise ValueError("Grid rows N must be at least 1.")
        _ncols = ncols if ncols is not None else N
        if _ncols < 1:
            raise ValueError("Grid cols ncols must be at least 1.")

        self.N     = N          # number of rows (kept for backward compatibility)
        self.nrows = N
        self.ncols = _ncols
        self.n_states  = self.nrows * self.ncols
        self.n_actions = len(self.ACTIONS)

        # Virtual absorbing state reached when boundary_terminal is triggered.
        # Its index (n_states) lies outside the normal state range so that
        # algorithms iterating over range(n_states) never process it, yet
        # is_terminal() returns True for it.
        self._boundary_state = self.n_states

        invalid = {s for s in terminal_states if not (0 <= s < self.n_states)}
        if invalid:
            raise ValueError(
                f"Terminal states out of range [0, {self.n_states - 1}]: {invalid}"
            )
        self.terminal_states = frozenset(terminal_states)

        self.stochastic        = stochastic
        self.slip_prob         = float(slip_prob)
        self.reward_fn         = reward_fn
        self.boundary_terminal = dict(boundary_terminal) if boundary_terminal else {}

        # Equiprobable random policy: π(a|s) = 1/n_actions for all s, a
        # Shape: (n_states, n_actions)
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move(self, state: int, action: int):
        """
        Attempt one grid step from *state* in *action* direction.

        Returns
        -------
        next_state : int   — new state index, or _boundary_state if OOB-terminal
        reward     : float
        done       : bool
        """
        row, col = divmod(state, self.ncols)
        dr, dc   = self._DELTAS[action]
        new_row, new_col = row + dr, col + dc

        if 0 <= new_row < self.nrows and 0 <= new_col < self.ncols:
            # Valid move inside the grid
            next_state = new_row * self.ncols + new_col
            done   = next_state in self.terminal_states
            reward = self._get_reward(state, action, next_state)
            return next_state, reward, done

        # Out of bounds
        if action in self.boundary_terminal:
            # Transition to the virtual absorbing terminal state
            return self._boundary_state, float(self.boundary_terminal[action]), True

        # Bounce — stay in place, still alive
        reward = self._get_reward(state, action, state)
        return state, reward, False

    def _get_reward(self, state: int, action: int, next_state: int) -> float:
        """Dispatch to custom reward_fn or return the default -1."""
        if self.reward_fn is not None:
            return float(self.reward_fn(state, action, next_state))
        return -1.0

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def get_transitions(self, state: int, action: int):
        """
        Return the transition distribution as a list of
        ``(probability, next_state, reward, done)`` tuples.

        Works for both deterministic and stochastic dynamics and is
        compatible with standard DP interfaces.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        if not self.stochastic:
            ns, r, d = self._move(state, action)
            return [(1.0, ns, r, d)]

        # Stochastic: aggregate probabilities over intended + perpendicular actions
        intended_prob = 1.0 - 2.0 * self.slip_prob
        candidates = [
            (action,                intended_prob),
            (self._PERP[action][0], self.slip_prob),
            (self._PERP[action][1], self.slip_prob),
        ]
        agg: dict = {}
        for act, prob in candidates:
            ns, r, d = self._move(state, act)
            key = (ns, r, d)
            agg[key] = agg.get(key, 0.0) + prob

        return [(prob, ns, r, d) for (ns, r, d), prob in agg.items()]

    def step(self, state: int, action: int):
        """
        Sample one transition from *state* under *action*.

        Returns
        -------
        next_state : int
        reward     : float  (0 if already terminal)
        done       : bool
        """
        if self.is_terminal(state):
            return state, 0.0, True

        transitions = self.get_transitions(state, action)
        if len(transitions) == 1:
            _, ns, r, d = transitions[0]
            return ns, r, d

        probs = [t[0] for t in transitions]
        idx   = np.random.choice(len(transitions), p=probs)
        _, ns, r, d = transitions[idx]
        return ns, r, d

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_episode(self, start_state: int = None, max_steps: int = 1000):
        """
        Roll out one episode following the current policy.

        Parameters
        ----------
        start_state : starting state index; if None, sampled uniformly from
                      all non-terminal in-grid states
        max_steps   : safety cap on episode length

        Returns
        -------
        episode : list of (state, action, reward) tuples
        """
        if start_state is None:
            non_terminal = [
                s for s in range(self.n_states) if not self.is_terminal(s)
            ]
            if not non_terminal:
                raise ValueError("All states are terminal; cannot start an episode.")
            start_state = int(np.random.choice(non_terminal))

        episode = []
        state   = start_state

        for _ in range(max_steps):
            if self.is_terminal(state):
                break
            action               = self.sample_action(state)
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        return episode

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    def sample_action(self, state: int) -> int:
        """Sample an action from the current policy at *state*."""
        return int(np.random.choice(self.n_actions, p=self.policy[state]))

    def set_policy(self, policy: np.ndarray):
        """
        Replace the policy table.

        Parameters
        ----------
        policy : array of shape (n_states, n_actions) with rows summing to 1
        """
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Policy shape must be ({self.n_states}, {self.n_actions}), "
                f"got {policy.shape}."
            )
        if not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("Each row of policy must sum to 1.")
        self.policy = policy

    def update_policy_greedy(self, Q: np.ndarray):
        """
        Set policy to be greedy with respect to *Q*.

        Parameters
        ----------
        Q : array of shape (n_states, n_actions)
        """
        for s in range(self.n_states):
            if not self.is_terminal(s):
                best_a = int(np.argmax(Q[s]))
                self.policy[s]        = 0.0
                self.policy[s][best_a] = 1.0

    def update_policy_epsilon_greedy(self, Q: np.ndarray, epsilon: float):
        """
        Set policy to ε-greedy with respect to *Q*.

        Parameters
        ----------
        Q       : array of shape (n_states, n_actions)
        epsilon : exploration probability ε ∈ [0, 1]
        """
        for s in range(self.n_states):
            if not self.is_terminal(s):
                best_a = int(np.argmax(Q[s]))
                self.policy[s]        = epsilon / self.n_actions
                self.policy[s][best_a] += 1.0 - epsilon

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def state_to_coords(self, state: int):
        """Convert a state index to (row, col)."""
        return divmod(state, self.ncols)

    def coords_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to a state index."""
        return row * self.ncols + col

    def is_terminal(self, state: int) -> bool:
        """True for in-grid terminal states and the virtual boundary-terminal state."""
        return state in self.terminal_states or state == self._boundary_state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, highlight_state: int = None):
        """Print an ASCII grid. 'T' = terminal, 'A' = agent, '.' = empty."""
        border = "+" + "---+" * self.ncols
        print(border)
        for row in range(self.nrows):
            row_cells = []
            for col in range(self.ncols):
                s = self.coords_to_state(row, col)
                if s == highlight_state:
                    row_cells.append(" A ")
                elif s in self.terminal_states:
                    row_cells.append(" T ")
                else:
                    row_cells.append(" . ")
            print("|" + "|".join(row_cells) + "|")
            print(border)

    def render_values(self, V, fmt: str = ".2f"):
        """
        Print the value function as an nrows × ncols grid.

        Parameters
        ----------
        V   : array-like of length n_states
        fmt : format specifier for float values (default '.2f')
        """
        V = np.asarray(V)
        for row in range(self.nrows):
            row_strs = []
            for col in range(self.ncols):
                s   = self.coords_to_state(row, col)
                val = float(V[s]) if s < len(V) else 0.0
                row_strs.append(f"{val:{fmt}}")
            print("  ".join(row_strs))

    def __repr__(self):
        shape = f"{self.nrows}×{self.ncols}"
        return (
            f"GridWorld({shape}, n_states={self.n_states}, "
            f"terminal_states={set(self.terminal_states)}, "
            f"stochastic={self.stochastic})"
        )

if __name__ == "__main__":
    # Example usage
    env = GridWorld(N=4, terminal_states={0, 15})
    print(env)
    env.render(highlight_state=5)

    state = 5
    action = 0  # up
    next_state, reward, done = env.step(state, action)
    print(f"From state {state} taking action '{env.ACTIONS[action]}':")
    print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

    # print the environment grid in command line multiple times to see the agent's movement
    for _ in range(5):
        env.render(highlight_state=state)
        action = env.sample_action(state)
        state, reward, done = env.step(state, action)
        print(f"Action taken: '{env.ACTIONS[action]}', Reward: {reward}, Done: {done}")
        if done:
            print("Reached terminal state. Resetting to start.")
            state = 5  # reset to a non-terminal state for demonstration
