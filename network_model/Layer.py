import numpy as np
from enum import Enum


class NeuronType(Enum):
    LIF_total = 1
    LIF_subtract = 2
    Hodgkin = 3

class STDP_FCLayer:
    def __init__(
        self,
        input_num,
        layer_size,
        n_type=NeuronType.LIF_total,
        is_first=False,
        k_winners=3,
        record_stdp=False,
        stdp_max_dt=50,
        record_state=False
    ):
        self.name = "STDP FC"
        self.n_type = n_type

        self.input_num = input_num
        self.layer_size = layer_size
        self.is_first = is_first
        self.k_winners = k_winners

        # -------------------------
        # Neuron state
        # -------------------------
        self.u = np.zeros(layer_size)

        self.weights = np.random.uniform(0.02, 0.15, size=(input_num, layer_size))

        # -------------------------
        # STDP parameters
        # -------------------------
        self.A_plus = 0.0005
        self.A_minus = 0.0008

        self.post_trace = np.zeros(layer_size)

        if self.is_first:
            self.pre_trace = np.zeros(input_num)

        # -------------------------
        # Competition / homeostasis
        # -------------------------
        self.threshold = np.ones(layer_size)
        self.winner_count = np.zeros(layer_size)

        self.threshold_increase = 0.05
        self.threshold_decay = 0.001

        # -------------------------
        # Optional state recording
        # -------------------------
        self.record_state = record_state

        self.spike_record = []
        self.mem_pot_record = []

        # -------------------------
        # Spike timing
        # -------------------------
        self.time = 0

        self.last_pre_spike = np.full(input_num, np.nan)

        self.last_post_spike = np.full(layer_size, np.nan)

        # -------------------------
        # Fixed-size STDP recording
        # -------------------------
        self.record_stdp = record_stdp
        self.stdp_max_dt = stdp_max_dt

        if self.record_stdp:
            num_bins = 2 * self.stdp_max_dt + 1

            # Sum of all dw values observed
            # for each delta-t
            self.stdp_dw_sum = np.zeros(num_bins, dtype=np.float64)

            # Number of events observed
            # for each delta-t
            self.stdp_count = np.zeros(num_bins, dtype=np.int64)

        self.plastic = True
        self.ltp_requests = np.zeros(layer_size)
        self.ltd_requests = np.zeros(layer_size)

    def __call__(self, inputs, pre_trace=None, target=None, learn=False, competitive=False):
        if len(inputs) != self.input_num:
            raise ValueError("The number of inputs is not equal to the number of neurons")

        inputs = np.asarray(inputs)

        # -------------------------
        # Decay traces
        # -------------------------
        self.post_trace *= 0.9

        if self.is_first:
            self.pre_trace *= 0.9

        # -------------------------
        # Forward pass
        # -------------------------
        new_mem_pot = np.dot(inputs, self.weights)

        spikes = self.update_u(new_mem_pot, competitive=competitive)

        # -------------------------
        # Teacher forcing on output
        # -------------------------
        if target is not None:
            learning_spikes = np.asarray(target)
        else:
            learning_spikes = spikes

        # -------------------------
        # Plasticity
        # -------------------------
        if learn and self.plastic:
            if self.is_first:
                self.update_weights(self.pre_trace, inputs, learning_spikes)
            elif target is not None:
                self.update_weights_supervised(pre_trace, spikes, learning_spikes, inputs)
            elif pre_trace is not None:
                self.update_weights(pre_trace, inputs, learning_spikes)
            else:
                raise ValueError("Non-first layer requires presynaptic trace")

        # Trace returned to next layer is
        # the OLD postsynaptic trace.
        ret_trace = self.post_trace.copy()

        # -------------------------
        # Update traces
        # -------------------------
        if target is not None:
            self.post_trace += target
        else:
            self.post_trace += spikes

        if self.is_first:
            self.pre_trace += inputs

        # -------------------------
        # Update spike timestamps
        #
        # Done AFTER plasticity so current
        # spikes don't count as previous spikes.
        # -------------------------
        pre_indices = np.flatnonzero(inputs == 1)

        self.last_pre_spike[pre_indices] = self.time

        post_indices = np.flatnonzero(learning_spikes == 1)

        self.last_post_spike[post_indices] = self.time

        self.time += 1

        return spikes, ret_trace

    def update_u(self, mem_pot_update, competitive=False):
        if self.n_type != NeuronType.LIF_total:
            raise NotImplementedError(f"{self.n_type} not implemented")

        # Add current
        self.u += mem_pot_update

        # =====================================
        # NORMAL LIF
        # =====================================
        if not competitive:
            spikes = (self.u >= self.threshold).astype(int)

            self.u[spikes == 1] = 0

            self.u = np.clip(self.u, 0, None)

            self.u *= 0.9

        # =====================================
        # k-WINNER-TAKE-ALL LIF
        # =====================================
        else:
            self.threshold += (1.0 - self.threshold) * self.threshold_decay

            candidate_idx = np.flatnonzero(self.u >= self.threshold)

            spikes = np.zeros(self.layer_size,dtype=int)

            if len(candidate_idx) > 0:
                k = min(self.k_winners, len(candidate_idx))

                # Get membrane potentials only
                # for neurons above threshold.
                candidate_u = self.u[candidate_idx]

                # Indices of the k largest
                # candidate membrane potentials.
                local_winners = np.argpartition(candidate_u, -k)[-k:]

                winner_idx = candidate_idx[local_winners]

                spikes[winner_idx] = 1

                # Homeostatic threshold increase
                self.threshold[winner_idx] += self.threshold_increase

                # Reset all neurons that crossed
                # threshold, including losers.
                self.u[candidate_idx] = 0

                self.winner_count[winner_idx] += 1

            self.u *= 0.9

        # -------------------------
        # Optional recording
        # -------------------------
        if self.record_state:
            self.spike_record.append(spikes.copy())

            self.mem_pot_record.append(self.u.copy())

        return spikes

    def update_weights(self, pre_trace, inputs, spikes):
        # -------------------------
        # Multiplicative STDP
        # -------------------------
        ltp = (self.A_plus * np.outer(pre_trace,spikes) * (1.0 - self.weights))

        ltd = (self.A_minus * np.outer(inputs, self.post_trace) * self.weights)

        dw = ltp - ltd

        # -------------------------
        # Record empirical STDP curve
        # -------------------------
        if self.record_stdp:
            self._record_stdp_events(inputs, spikes, ltp, ltd)

        # -------------------------
        # Apply update
        # -------------------------
        self.weights += dw

        self.weights = np.clip(self.weights, 0.0, 1.0)

    def update_weights_supervised(self, pre_trace, target_spikes, actual_spikes, inputs):
        error = target_spikes - actual_spikes
        neg_error = np.maximum(0, -error)
        pos_error = np.maximum(0, error)

        self.ltp_requests += pos_error
        self.ltd_requests += neg_error

        effective_pretrace = pre_trace + inputs

        ltp = self.A_plus * np.outer(effective_pretrace, pos_error) * (1.0 - self.weights)

        ltd = self.A_minus * np.outer(effective_pretrace, neg_error) * self.weights

        self.weights += ltp - ltd

        self.weights = np.clip(self.weights, 0.0, 1.0)

    def _record_stdp_events(self, inputs, spikes, ltp, ltd):
        """
        Collect empirical delta-w as a
        function of delta-t without storing
        every event individually.

        Convention:

            dt = t_post - t_pre

        Therefore:
            dt > 0 -> LTP
            dt < 0 -> LTD
        """

        # =====================================
        # LTP
        # Current post spike paired with
        # most recent previous pre spike.
        # =====================================
        post_idx = np.flatnonzero(spikes == 1)

        for j in post_idx:
            valid_pre = np.flatnonzero(~np.isnan(self.last_pre_spike))

            for i in valid_pre:
                if ltp[i, j] == 0:
                    continue

                dt = int(self.time - self.last_pre_spike[i])

                self.record_stdp_event(dt, ltp[i, j])

        # =====================================
        # LTD
        # Current pre spike paired with
        # most recent previous post spike.
        # =====================================
        pre_idx = np.flatnonzero(inputs == 1)

        valid_post = np.flatnonzero(~np.isnan(self.last_post_spike))

        for i in pre_idx:
            for j in valid_post:
                if ltd[i, j] == 0:
                    continue

                dt = int(self.last_post_spike[j] - self.time)

                self.record_stdp_event(dt, -ltd[i, j])

    def record_stdp_event(self, dt, dw):
        """
        Add one STDP observation to its
        delta-t bin.

        Memory usage remains constant.
        """

        if -self.stdp_max_dt <= dt <= self.stdp_max_dt:
            idx = dt + self.stdp_max_dt

            self.stdp_dw_sum[idx] += float(dw)

            self.stdp_count[idx] += 1

    def get_stdp_curve(self):
        """
        Returns:

            dt
            mean_dw
            count

        ready for plotting.
        """

        if not self.record_stdp:
            raise RuntimeError("STDP recording was not enabled for this layer.")

        dt = np.arange(-self.stdp_max_dt, self.stdp_max_dt + 1)

        mean_dw = np.divide(
            self.stdp_dw_sum,
            self.stdp_count,
            out=np.zeros_like(self.stdp_dw_sum),
            where=self.stdp_count > 0
        )

        return dt, mean_dw, self.stdp_count.copy()

    def reset_state(self):
        self.u[:] = 0
        self.post_trace[:] = 0

        if self.is_first:
            self.pre_trace[:] = 0

        self.time = 0

        self.last_pre_spike[:] = np.nan
        self.last_post_spike[:] = np.nan

    def reset_winner_count(self):
        self.winner_count[:] = 0

    def reset_stdp_record(self):
        """
        Clear accumulated STDP statistics
        without changing network weights.
        """

        if self.record_stdp:
            self.stdp_dw_sum[:] = 0
            self.stdp_count[:] = 0

    def clear_state_records(self):
        self.spike_record.clear()
        self.mem_pot_record.clear()

    def set_plastic(self, plastic: bool):
        self.plastic = plastic

    def reset_request(self):
        self.ltp_requests[:] = 0
        self.ltd_requests[:] = 0

    def __str__(self):
        return f"{self.name}: {self.input_num} in -> {self.layer_size} out"

    def __repr__(self):
        return self.__str__()

class CMAES_FCLayer:
    def __init__(self, input_num: int, layer_size: int, record_state: bool = False, n_type: NeuronType = NeuronType.LIF_total, decay: float = 0.9):
        self.name = "CMAES FC"

        self.input_num = input_num
        self.layer_size = layer_size

        self.u = np.zeros(layer_size)
        self.weights = np.random.uniform(0.02, 0.15, size=(input_num, layer_size))
        self.threshold = 1.0

        self.record_state = record_state

        self.spike_record = []
        self.mem_pot_record = []

        self.n_type = n_type

        self.learning = True

        self.decay = decay

    def __call__(self, inputs):
        if len(inputs) != self.input_num:
            raise ValueError("The number of inputs is not equal to the number of neurons")

        inputs = np.asarray(inputs)

        new_mem_pot = np.dot(inputs, self.weights)

        spikes = self.update_u(new_mem_pot)

        if self.record_state:
            self.spike_record.append(spikes.copy())
            self.mem_pot_record.append(self.u.copy())

        return spikes

    def update_u(self, mem_pot_update):
        if self.n_type != NeuronType.LIF_total:
            raise NotImplementedError(f"{self.n_type} not implemented")

        self.u += mem_pot_update

        spikes = (self.u >= self.threshold).astype(int)

        self.u[spikes == 1] = 0

        self.u = np.clip(self.u, 0, None)

        self.u *= self.decay

        return spikes

    def get_params(self):
        return self.weights

    def set_params(self, theta):
        self.weights = theta.reshape(self.input_num, self.layer_size)

    def reset_state(self):
        self.u[:] = 0

    def __str__(self):
        return f"{self.name}: {self.input_num} in -> {self.layer_size} out"

    def __repr__(self):
        return self.__str__()

class EGGROLL_FCLayer(CMAES_FCLayer):
    """LIF FC layer with a batched, factorised EGGROLL population forward pass.

    Single-sample calls exactly inherit CMAES_FCLayer's behaviour, including
    weights/get_params/set_params/reset_state. Population training instead calls
    population_step with inputs (population, input_num), membrane state
    (population, layer_size), and optional rank-r factors (A, B), with:

        current = inputs @ weights + sigma / sqrt(rank) * (inputs @ A) @ B.T

    Crucially no P x input_num x layer_size dense perturbations are allocated.
    """
    def __init__(self, input_num: int, layer_size: int, record_state=False,
                 n_type=NeuronType.LIF_total, decay=0.9):
        super().__init__(input_num, layer_size, record_state=record_state,
                         n_type=n_type, decay=decay)
        self.name = "EGGROLL FC"

    def population_step(self, inputs, membrane, factors=None, sigma=0.):
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_num:
            raise ValueError(f"Expected population inputs (P, {self.input_num})")
        if membrane.shape != (inputs.shape[0], self.layer_size):
            raise ValueError(f"Expected membrane {(inputs.shape[0], self.layer_size)}")
        if self.n_type != NeuronType.LIF_total:
            raise NotImplementedError(f"{self.n_type} not implemented")
        current = inputs @ self.weights  # shared dense weights across all candidates
        if factors is not None:
            a, b = factors
            if a.shape[:2] != (len(inputs), self.input_num) or b.shape[:2] != (len(inputs), self.layer_size):
                raise ValueError("Low-rank population factors have wrong dimensions")
            rank = a.shape[-1]
            if rank < 1 or b.shape[-1] != rank:
                raise ValueError("Rank must match and be >= 1")
            projected = np.einsum("pi,pir->pr", inputs, a, optimize=True)
            current += (sigma / np.sqrt(rank)) * np.einsum("pr,por->po", projected, b, optimize=True)
        membrane += current
        spikes = (membrane >= self.threshold).astype(np.int64)
        membrane[spikes != 0] = 0.
        np.maximum(membrane, 0., out=membrane)
        membrane *= self.decay
        return spikes
