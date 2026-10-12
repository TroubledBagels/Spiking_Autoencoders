import network_model.Layer as L
import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def __str__(self):
        string = ""
        for i, layer in enumerate(self.layers):
            string += f"{i+1}: " + str(layer) + "\n"

        return string

    def __call__(self, inputs, target=None, learn=False):
        trace = None
        for i, layer in enumerate(self.layers):
            # CMAES_FCLayer and EGGROLL_FCLayer return just spikes, whereas
            # STDP_FCLayer returns (spikes, trace). Keep both APIs compatible.
            if isinstance(layer, L.CMAES_FCLayer):
                inputs = layer(inputs)
                trace = None
            elif i == len(self.layers) - 1:
                inputs, trace = layer(inputs, trace, target, learn=learn, competitive=False)
            else:
                inputs, trace = layer(inputs, trace, learn=learn, competitive=True)
        return inputs

    def get_spike_records(self):
        spike_records = []
        for layer in self.layers:
            spike_records.append(layer.spike_record)

        return spike_records

    def get_mem_pot_records(self):
        mem_pot_records = []
        for layer in self.layers:
            mem_pot_records.append(layer.mem_pot_record)

        return mem_pot_records


def interpret_records(spike_records: list):
    i = 1
    for record in spike_records:
        print(f"Layer {i}:")
        for t in record:
            for u in t:
                if u % 1 == 0:
                    print(f"{u} ", end='')
                else:
                    print(f"{u:.2f} ", end='')
            print()
        i += 1


if __name__ == "__main__":
    network = Network()
    network.add(L.STDP_FCLayer(2, 4, k_winners=10))
    network.add(L.STDP_FCLayer(4, 6))

    test_inp = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])

    print(network)

    for inp in test_inp:
        network(inp)

    interpret_records(network.get_spike_records())

    interpret_records(network.get_mem_pot_records())