# Pure Python version of madmom.ml.hmm (ported from Cython)
import numpy as np
import warnings
INFINITY = 1e20
class TransitionModel:
    def __init__(self, states, pointers, probabilities):
        self.states = states
        self.pointers = pointers
        self.probabilities = probabilities
    @property
    def num_states(self):
        return len(self.pointers) - 1
    @property
    def num_transitions(self):
        return len(self.probabilities)
    @property
    def log_probabilities(self):
        return np.log(self.probabilities)
    @staticmethod
    def make_dense(states, pointers, probabilities):
        from scipy.sparse import csr_matrix
        transitions = csr_matrix((np.array(probabilities),
                                  np.array(states), np.array(pointers)))
        states, prev_states = transitions.nonzero()
        return states, prev_states, probabilities
    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        from scipy.sparse import csr_matrix
        states = np.asarray(states)
        prev_states = np.asarray(prev_states, dtype=int)
        probabilities = np.asarray(probabilities)
        if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
            raise ValueError('Not a probability distribution.')
        num_states = max(prev_states) + 1
        transitions = csr_matrix((probabilities, (states, prev_states)),
                                 shape=(num_states, num_states))
        return transitions.indices.astype(np.uint32), \
               transitions.indptr.astype(np.uint32), \
               transitions.data.astype(float)
    @classmethod
    def from_dense(cls, states, prev_states, probabilities):
        transitions = cls.make_sparse(states, prev_states, probabilities)
        return cls(*transitions)
class ObservationModel:
    def __init__(self, pointers):
        self.pointers = pointers
    def log_densities(self, observations):
        raise NotImplementedError
    def densities(self, observations):
        return np.exp(self.log_densities(observations))
class DiscreteObservationModel(ObservationModel):
    def __init__(self, observation_probabilities):
        if not np.allclose(observation_probabilities.sum(axis=1), 1):
            raise ValueError('Not a probability distribution.')
        super().__init__(np.arange(observation_probabilities.shape[0], dtype=np.uint32))
        self.observation_probabilities = observation_probabilities
    def densities(self, observations):
        return self.observation_probabilities[:, observations].T
    def log_densities(self, observations):
        return np.log(self.densities(observations))
class HiddenMarkovModel:
    def __init__(self, transition_model, observation_model, initial_distribution=None):
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = np.ones(transition_model.num_states) / transition_model.num_states
        if not np.allclose(initial_distribution.sum(), 1):
            raise ValueError('Initial distribution is not a probability distribution.')
        self.initial_distribution = initial_distribution
        self._prev = self.initial_distribution.copy()
    def reset(self, initial_distribution=None):
        self._prev = initial_distribution or self.initial_distribution.copy()
    def viterbi(self, observations):
        tm = self.transition_model
        om = self.observation_model
        num_states = tm.num_states
        num_observations = len(observations)
        current_viterbi = np.empty(num_states, dtype=float)
        previous_viterbi = np.log(self.initial_distribution)
        bt_pointers = np.empty((num_observations, num_states), dtype=np.uint32)
        om_densities = om.log_densities(observations)
        om_ptrs = om.pointers
        tm_ptrs = tm.pointers
        tm_states = tm.states
        tm_probs = tm.log_probabilities
        for frame in range(num_observations):
            for state in range(num_states):
                density = om_densities[frame, om_ptrs[state]]
                best_score = -INFINITY
                best_prev = 0
                for pointer in range(tm_ptrs[state], tm_ptrs[state + 1]):
                    prev_state = tm_states[pointer]
                    score = previous_viterbi[prev_state] + tm_probs[pointer] + density
                    if score > best_score:
                        best_score = score
                        best_prev = prev_state
                current_viterbi[state] = best_score
                bt_pointers[frame, state] = best_prev
            previous_viterbi[:] = current_viterbi[:]
        final_state = np.argmax(current_viterbi)
        log_prob = current_viterbi[final_state]
        if np.isinf(log_prob):
            warnings.warn('-inf log probability during Viterbi decoding')
            return np.empty(0, dtype=np.uint32), log_prob
        path = np.empty(num_observations, dtype=np.uint32)
        for frame in range(num_observations - 1, -1, -1):
            path[frame] = final_state
            final_state = bt_pointers[frame, final_state]
        return path, log_prob
    def forward(self, observations, reset=True):
        tm = self.transition_model
        om = self.observation_model
        num_states = tm.num_states
        num_observations = len(observations)
        if reset:
            self.reset()
        fwd = np.zeros((num_observations, num_states), dtype=float)
        fwd_prev = self._prev
        om_densities = om.densities(observations)
        om_ptrs = om.pointers
        tm_ptrs = tm.pointers
        tm_states = tm.states
        tm_probs = tm.probabilities
        for frame in range(num_observations):
            prob_sum = 0
            for state in range(num_states):
                total = 0
                for ptr in range(tm_ptrs[state], tm_ptrs[state + 1]):
                    total += fwd_prev[tm_states[ptr]] * tm_probs[ptr]
                fwd[frame, state] = total * om_densities[frame, om_ptrs[state]]
                prob_sum += fwd[frame, state]
            norm_factor = 1. / prob_sum
            fwd[frame] *= norm_factor
            fwd_prev = fwd[frame].copy()
        return fwd
HMM = HiddenMarkovModel
