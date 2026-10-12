import EDDataset as edd
import pathlib
from torchvision import transforms
import torchaudio
import tqdm
import torch.utils
import network_model.Network as N
import network_model.Layer as L
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def train(model: N.Network, train_loader, test_loader, n_epochs=100, delay=0):

    for epoch in range(n_epochs):
        if epoch == 4:
            model.layers[0].set_plastic(False)

        weight_history = []
        model.layers[0].reset_winner_count()

        pbar = tqdm.tqdm(train_loader)

        for idx, (inputs, _) in enumerate(pbar):
            inputs = inputs.detach().numpy()[0]
            targets = inputs.copy()

            model.reset_state()

            for t in range(inputs.shape[0]):
                input_spikes = inputs[t]
                target_spikes = targets[t]
                _ = model(input_spikes, target_spikes, learn=True)

            if idx % 50 == 0:
                weight_history.append(model.layers[-1].weights.mean(axis=0).copy())

        weight_history = np.array(weight_history)

        for j in range(model.layers[-1].layer_size):
            plt.plot(weight_history[:, j], label=f"Out {j}")

        plt.xlabel("Checkpoint")
        plt.ylabel("Mean decoder weight")
        plt.legend()
        plt.show()

        model.layers[0].reset_winner_count()

        tp = 0
        fp = 0
        fn = 0
        out_spike_count = 0
        out_total = 0
        target_spike_count = 0
        target_total = 0

        lags = range(-5, 6)
        lag_stats = {lag: {"tp": 0, "fp": 0, "fn": 0} for lag in lags}

        qbar = tqdm.tqdm(test_loader)

        channel_tp = np.zeros(model.layers[0].input_num)
        channel_fp = np.zeros(model.layers[0].input_num)
        channel_fn = np.zeros(model.layers[0].input_num)
        channel_target_spikes = np.zeros(model.layers[0].input_num)
        channel_output_spikes = np.zeros(model.layers[0].input_num)

        for idx, (inputs, _) in enumerate(qbar):
            inputs = inputs.detach().numpy()[0]
            target = inputs.copy()

            model.reset_state()

            outputs = []

            for t in range(inputs.shape[0]):
                input_spikes = inputs[t]
                output = model(input_spikes)

                out_total += output.size

                outputs.append(output.copy())

                channel_target_spikes += target[t]
                channel_output_spikes += output

            outputs = np.array(outputs)

            target_spike_count += target.sum()
            target_total += target.size

            target_bool = target.astype(bool)
            output_bool = outputs.astype(bool)

            tp += np.logical_and(target_bool, output_bool).sum()
            fp += np.logical_and(~target_bool, output_bool).sum()
            fn += np.logical_and(target_bool, ~output_bool).sum()

            channel_tp += np.logical_and(target_bool, output_bool).sum(axis=0)
            channel_fp += np.logical_and(~target_bool, output_bool).sum(axis=0)
            channel_fn += np.logical_and(target_bool, ~output_bool).sum(axis=0)

            for lag in lags:
                if lag == 0:
                    lag_target = target
                    lag_output = outputs
                elif lag > 0:
                    lag_target = target[:-lag]
                    lag_output = outputs[lag:]
                else:
                    d = -lag
                    lag_target = target[d:]
                    lag_output = outputs[:-d]

                lag_target_bool = lag_target.astype(bool)
                lag_output_bool = lag_output.astype(bool)

                lag_stats[lag]["tp"] += np.logical_and(lag_target_bool, lag_output_bool).sum()
                lag_stats[lag]["fp"] += np.logical_and(~lag_target_bool, lag_output_bool).sum()
                lag_stats[lag]["fn"] += np.logical_and(lag_target_bool, ~lag_output_bool).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        output_firing_rate = out_spike_count / out_total
        target_firing_rate = target_spike_count / target_total

        print(f"Epoch {epoch + 1}/{n_epochs} completed.")
        print(f"Prec: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print("Layer 1 Test Winner Count:")
        print(model.layers[0].winner_count)
        print("Layer 2 Test Firing Rate/Target Firing Rate:")
        print(f"{output_firing_rate:.4f} vs {target_firing_rate:.4f}")

        print("\nF1 by temporal lag:")

        best_lag = None
        best_f1 = -1

        for lag in lags:
            lag_tp = lag_stats[lag]["tp"]
            lag_fp = lag_stats[lag]["fp"]
            lag_fn = lag_stats[lag]["fn"]

            lag_precision = lag_tp / (lag_tp + lag_fp) if lag_tp + lag_fp > 0 else 0.0
            lag_recall = lag_tp / (lag_tp + lag_fn) if lag_tp + lag_fn > 0 else 0.0
            lag_f1 = 2 * lag_precision * lag_recall / (lag_precision + lag_recall) if lag_precision + lag_recall > 0 else 0.0

            print(f"Lag {lag:+d}: P={lag_precision:.4f}, R={lag_recall:.4f}, F1={lag_f1:.4f}")

            if lag_f1 > best_f1:
                best_f1 = lag_f1
                best_lag = lag

        print(f"Best lag: {best_lag:+d}, F1={best_f1:.4f}")

        for i in range(10):
            p = channel_tp[i] / (channel_tp[i] + channel_fp[i]) if channel_tp[i] + channel_fp[i] > 0 else 0
            r = channel_tp[i] / (channel_tp[i] + channel_fn[i]) if channel_tp[i] + channel_fn[i] > 0 else 0
            f = 2 * p * r / (p + r) if p + r > 0 else 0
            print(f"Channel {i}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")

        print(f"Channel Target Spike Number: {channel_target_spikes}")
        print(f"Channel Output Spike Number: {channel_output_spikes}")

        print(f"Encoder weights:")
        print(np.round(model.layers[0].weights, 2))
        print(f"Output layer weights:")
        print(np.round(model.layers[1].weights, 2))

        print("Decoder LTP requests:")
        print(model.layers[-1].ltp_requests)

        print("Decoder LTD requests:")
        print(model.layers[-1].ltd_requests)

        model.layers[-1].reset_request()

def calculate_f1(target, output):
    target = target.astype(bool)
    output = output.astype(bool)

    tp = np.logical_and(target, output).sum()

    fp = np.logical_and(~target, output).sum()

    fn = np.logical_and(target, ~output).sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0

    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":
    home = pathlib.Path("~").expanduser()
    local_dir = home / "data" / "URBAN-SED"

    DELAY = 2

    a_transform = transforms.Compose([
        torchaudio.transforms.Resample(44100, 16000),
        edd.UnsqueezeTransform(dim=0),
        edd.ToSpikeTransform(num_channels=5),
        edd.SqueezeTransform(dim=0),
        edd.SqueezeTransform(dim=0),
    ])

    tr_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform)
    te_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=1, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=True)

    model = N.Network()
    model.add([
        L.STDP_FCLayer(input_num=10, layer_size=10, k_winners=7, is_first=True),
        # L.FCLayer(input_num=60, layer_size=60),
        L.STDP_FCLayer(input_num=10, layer_size=10)
    ])

    train(model, tr_dl, te_dl, delay=DELAY)