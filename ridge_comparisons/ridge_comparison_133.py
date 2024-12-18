"""
Author: Marcus Jaiclin (TBSDrJ), The Bishop's School, in collobaration with 
    Virginia DeSa and Simon Fei (simonfei123), UCSD.
Date: Nov/Dec 2024
N.B.: Folder paths in this program assume it is being run from the *parent*
    folder to the folder it is in.
"""
import pickle
import time
import argparse
import pathlib

import numpy as np
from scipy.spatial.distance import correlation
import torch
import sklearn.linear_model

# Constants for use in filenames so that they all match.
VERSION = __file__.split(".")[0].split("_")[-1]
TIME_STAMP = time.strftime('%Y_%m_%d_%H_%M')

def load_eeg_data(
        duration: int, sub: int, train_size: int, average: str
        ) -> (np.ndarray, np.ndarray):
    """EEG data is inputs for the model"""
    print(f"Loading EEG Data for subject {sub}, duration {duration}ms")
    if duration != 0:
        eeg_train = np.load(f"data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy")[:train_size,:,:duration]
        eeg_test = np.load(f"data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg{average}.npy")[:,:,:duration]
    else:
        eeg_train = np.load(f"data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg_null.npy")[:train_size]
        eeg_test = np.load(f"data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg{average}_null.npy")
    print(f"{eeg_train.shape=}, {eeg_test.shape=}")
    return eeg_train, eeg_test

def preprocess_eeg_data(eeg_train: np.ndarray, eeg_test: np.ndarray
        ) -> (np.ndarray, np.ndarray):
    """Convert to float32, then z-score. Shape will vary for some models.
    
    In many cases, we flatten the 17xduration values, but in some cases, we
    will treat the EEG channels differently from the time channels."""
    print("Preprocessing EEG data")
    eeg_train = eeg_train.astype(np.float32)
    eeg_test = eeg_test.astype(np.float32)
    norm_mean_train = np.mean(eeg_train, axis=0)
    norm_scale_train = np.std(eeg_train, axis=0, ddof=1)
    eeg_train = (eeg_train - norm_mean_train) / norm_scale_train
    eeg_test = (eeg_test - norm_mean_train) / norm_scale_train
    # Flatten EEG channels and time dim into one dimension
    eeg_train = eeg_train.reshape(eeg_train.shape[0],-1)
    eeg_test = eeg_test.reshape(eeg_test.shape[0],-1)
    print(f"{eeg_train.shape=}, {eeg_test.shape=}")
    return eeg_train, eeg_test    

def load_latents(sz: int) -> (np.ndarray, np.ndarray):
    """Latents are the targets in the training process."""
    print("Load latents")
    ids = list(range(sz))
    train_latents= np.load("cache/thingseeg2_extracted_embeddings/train_autokl.npy", mmap_mode="r")[ids]
    test_latents = np.load("cache/thingseeg2_extracted_embeddings/test_autokl.npy", mmap_mode="r")
    print(f"{train_latents.shape=} {test_latents.shape=}")
    return train_latents, test_latents

def save_code() -> None:
    """Save a copy of the code to keep track of what we've tried."""
    with open(__file__, "r") as f:
        lines = f.readlines()
    with open(f"ridge_comparisons/saves/save_code_{VERSION}_" + 
            f"{TIME_STAMP}.py", "w") as f:
        for line in lines:
            print(line[:-1], file=f)

def prefill_model_coeffs(
            model: torch.nn.Module,
            eeg_train: np.ndarray,
            eeg_test: np.ndarray,
            train_latents: np.ndarray,
            test_latents: np.ndarray,
    ) -> torch.nn.Module:
    """Train the Ridge regression, copy weights into the torch layer."""
    reg = sklearn.linear_model.Ridge(
            alpha=1000, max_iter=50000, fit_intercept=True)
    reg.fit(eeg_train, train_latents)
    print(f"Ridge coeffs: {reg.coef_.shape}")
    wgt_param = torch.nn.Parameter(
            data=torch.tensor(reg.coef_.copy(), dtype=torch.float32))
    bias_param = torch.nn.Parameter(
            data=torch.tensor(reg.intercept_.copy(), dtype=torch.float32))
    print(f"Linear layer coeffs: {model.linear_ridge.weight.shape}")
    model.linear_ridge.weight = wgt_param
    model.linear_ridge.bias = bias_param
    avg_euclidean_distance, avg_correlation = evaluate(
            eeg_test, train_latents, test_latents, reg)
    print(f"Evaluate Ridge: {avg_euclidean_distance=}, {avg_correlation=}")
    return model

def train_model(
            model: torch.nn.Module, 
            train: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, 
            lr_sch: torch.optim.lr_scheduler.LRScheduler | None,
            my_loss: torch.nn.Module,
            eeg_test: np.ndarray, 
            train_latents: np.ndarray,
            test_latents: np.ndarray, 
    ) -> None:
    """Run training loop."""
    filename = (f"ridge_comparisons/saves/save_printout_{VERSION}_" + 
            f"{TIME_STAMP}.txt")
    stopping = False
    epoch_losses = []
    distances = []
    correlations = []
    improvements = []
    epoch = 1
    avg_euclidean_distance, avg_correlation = evaluate(
            eeg_test, train_latents, test_latents, model)
    distances.append(avg_euclidean_distance)
    correlations.append(avg_correlation)
    print(f"Evaluate model before start: {avg_euclidean_distance=:.2f}, " +
            f"{avg_correlation=:.6f}")
    with open(filename, "a") as f:
        print (f"Evaluate model before start: {avg_euclidean_distance=:.2f}, " + 
                f"{avg_correlation=:.6f}\n", file=f)
    while not stopping:
        batch_losses = []
        batch = 0
        start = time.perf_counter()
        output_str = ""
        for (train_batch, latent_batch) in train:
            pred = model(train_batch)
            loss = my_loss(pred, latent_batch, model)
            # Loss object is *much* larger than just the decimal value
            batch_losses.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch += 1
            print(f"Epoch {epoch:03} Batch {batch:03} " + 
                    f"Batch train loss: {batch_losses[-1]:.2f}\r",
                    end="", flush=True)
            output_str += (f"Epoch {epoch:03} Batch {batch:03} " + 
                    f"Batch train loss: {batch_losses[-1]:.4f}\n")
        if lr_sch is not None:
            lr_sch.step()
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        print(f"\nEpoch avg. loss: {epoch_losses[-1]:.2f}")
        output_str += f"Epoch avg. loss: {epoch_losses[-1]:.2f}\n"
        # If metric does not improve over prior minimum for 10 epochs, stop.
        avg_euclidean_distance, avg_correlation = evaluate(
                eeg_test, train_latents, test_latents, model)
        if epoch > 1:
            improvement = ((min(distances) - avg_euclidean_distance)
                    /min(distances))
            improvement += ((avg_correlation - max(correlations))
                    /max(correlations))
            improvements.append(improvement)
            print(f"% Improvement over best so far: {improvement * 100:.4f}")
            output_str += ("% Improvement over best so far: " + 
                    f"{improvement * 100:.4f}\n")
        distances.append(avg_euclidean_distance)
        correlations.append(avg_correlation)
        print(f"{avg_euclidean_distance=:.2f}, {avg_correlation=:.6f}")
        output_str += f"{avg_euclidean_distance=:.2f}, {avg_correlation=:.6f}\n"
        print(f"Min dist so far: {min(distances):.2f}, Max corr so far: " + 
                f"{max(correlations):.8f}")
        output_str += (f"Min dist so far: {min(distances):.2f}, Max corr " + 
                f"so far: {max(correlations):.8f}\n")
        if epoch < 11: 
            stopping = False
            print("Epoch count less than 11, continue training.")
        else:
            test = sum([im > 0 for im in improvements[-10:]])
            if test == 0:
                stopping = True
                print("Failed to improve in 10 consecutive " + 
                        "epochs, stop.")
                output_str += ("Failed to improve in 10 " +
                        "consecutive epochs, stop.\n")
            else:
                stopping = False
                print(f"Improved in {test} of the last " + 
                        "10 epochs, continue training.")
                output_str += (f"Improved in {test} of " + 
                        "the last 10 epochs, continue training.\n")
        if avg_euclidean_distance < 113.3732 or avg_correlation > 0.023811:
            with open(f"ridge_comparisons/saves/models/save_model_{VERSION}_" + 
                    f"{TIME_STAMP}_epoch_{epoch}.dat", "wb") as f:
                pickle.dump(model, f)
        print(f"Time: {int(time.perf_counter() - start)} secs\n")
        output_str += f"Time: {int(time.perf_counter() - start)} secs\n"
        with open(filename, "a") as f:
            print(output_str, file=f)
        epoch += 1


def evaluate(eeg_test: np.ndarray, train_latents: np.ndarray,
        test_latents: np.ndarray, model: torch.nn.Module) -> (float, float):
    """Compute predicted latents from test EEG data, compare to actual"""
    if isinstance(model, MyModel):
        eeg_test = torch.tensor(eeg_test)
        with torch.no_grad():
            pred_latent = model(eeg_test)
        pred_latent = pred_latent.to("cpu").detach().numpy()
    else:
        pred_latent = model.predict(eeg_test) 
    pred_latent_mean = np.mean(pred_latent,axis=0)
    pred_latent_std = np.std(pred_latent,axis=0)
    std_norm_pred_latent = (pred_latent - pred_latent_mean) / pred_latent_std
    train_latents_mean = np.mean(train_latents,axis=0)
    train_latents_std = np.std(train_latents,axis=0)
    pred_latents = std_norm_pred_latent * train_latents_std + train_latents_mean
    euclidean_distances = np.array([np.linalg.norm(u - v) 
            for u, v in zip(pred_latents, test_latents)])
    correlation_distances = np.array([correlation(u, v) 
            for u, v in zip(pred_latents, test_latents)])
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()
    return average_euclidean_distance, correlations

class MyZipDataset(torch.utils.data.Dataset):
    """Build Torch Dataset from the loaded EEG data"""
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets
        self.length = inputs.shape[0]
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, i: int) -> (torch.Tensor, torch.Tensor):
        dtype = torch.float32
        inp = torch.tensor(self.inputs[i], dtype=dtype)
        target = torch.tensor(self.targets[i], dtype=dtype)
        return inp, target

class MyLoss(torch.nn.Module):
    """Loss functions will vary from version to version."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, 
            model: torch.nn.Module, α=1, β=1) -> torch.Tensor:
        loss_0 = torch.nn.MSELoss(reduction="sum")(preds, targets)
        wgts_0 = model.get_parameter('linear_ridge.weight')
        bias_0 = model.get_parameter('linear_ridge.bias')
        loss_1 = wgts_0.abs().sum() + bias_0.abs().sum()
        loss_2 = 0
        return loss_0 + α*loss_1 + β*loss_2

class MyModel(torch.nn.Module):
    """The actual model that we will train which will vary version to version"""
    def __init__(self, duration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Using 17 sensors on the EEG headset
        input_size = 17 * duration
        self.linear_ridge = torch.nn.Linear(input_size, 91168)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.linear_ridge(x)
        return outputs

def main():
    DURATION = 40
    SUBJECT = 1
    TRAIN_SIZE = 16540
    AVERAGE = ""

    torch.manual_seed(0)
    torch.set_default_device(torch.device("mps"))

    save_code()

    # Setup based on settings
    eeg_train, eeg_test = load_eeg_data(DURATION, SUBJECT, TRAIN_SIZE, AVERAGE)
    eeg_train, eeg_test = preprocess_eeg_data(eeg_train, eeg_test)
    train_latents, test_latents = load_latents(len(eeg_train))
    train_dataset = MyZipDataset(eeg_train, train_latents)
    train = torch.utils.data.DataLoader(train_dataset, 
            batch_size=64, drop_last=True)
    model = MyModel(duration = DURATION)
    model = prefill_model_coeffs(model, eeg_train, eeg_test, train_latents, test_latents)
    avg_euclidean_distance, avg_correlation = evaluate(
            eeg_test, train_latents, test_latents, model)
    print(f"{avg_euclidean_distance=:.2f}, {avg_correlation=:.6f}")
    model.train()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda epoch: 0.99)
    loss = MyLoss()
    print(f"{model=}")
    print("Training VDVAE Regression")
    # Train loop evaluates at every epoch using same metrics used on Ridge
    train_model(model, train, optimizer, lr_sch, loss, 
            eeg_test, train_latents, test_latents)
    print("VDVAE training complete")

if __name__ == "__main__":
    main()
