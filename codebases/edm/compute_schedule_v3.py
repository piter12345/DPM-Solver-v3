"""Training and evaluation"""

from absl import app
from absl import flags
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time
from edm_base import EDM

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("n_batch", 1, "Number of batches per GPU", lower_bound=1)
flags.DEFINE_integer("batch_size", 512, "Batch size per GPU", lower_bound=1)
flags.DEFINE_integer("n_timesteps", 1200, "Number of timesteps", lower_bound=1)
flags.DEFINE_string("ckp_path", None, "Checkpoint path")

flags.mark_flags_as_required(["ckp_path", "workdir"])

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_dataset_multi_host(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True
    )

    # print(f"Load dataset slice {slice}/{num_slices}, trainset length {len(train_ds)}, evalset length {len(eval_ds)}")
    return trainloader


def elbo(x_t, x_0, x_hat):
    mse = ((x_0 - x_hat) ** 2).mean()
    return mse

def compute_elbos(
    framework, statistics_dir, MAX_BATCH, n_timesteps, batch_size, device
):

    print(f"compute_elbos")
    trainloader, _, _ = get_dataset_multi_host(framework.data, batch_size)

    ns = framework.noise_schedule
    timesteps = framework.get_timesteps(n_timesteps, device)
    framework.create_model(device)
    model_fn = framework.model_fn

    if os.path.exists(os.path.join(statistics_dir, f"elbos.npz")):
        return
    elbos_lst = [0] * len(timesteps)
    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing elbos..."):
            time_start = time.time()
            for i, (batch, _) in enumerate(trainloader):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = batch.to(device).float()
                train_batch = train_batch.permute(0, 3, 1, 2)
                x = train_batch

                v = torch.randint(0, 2, x.shape, device=device) * 2.0 - 1
                z = torch.randn_like(x)
                alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
                perturbed_data = alpha_t * x + sigma_t * z

                x_hat = model_fn(perturbed_data, t)
                
                elbos_lst[j] += elbo(perturbed_data, x, x_hat)
            elbos_lst[j] = elbos_lst[j] / MAX_BATCH
    elbos_lst = np.asarray(elbos_lst)
    # np.savez_compressed(os.path.join(statistics_dir, f"elbos_{r}.npz"), elbos=elbos_lst)
    np.savez_compressed(os.path.join(statistics_dir, f"elbos.npz"), elbos=elbos_lst)


def collect_elbos(statistics_dir):
    print("Collecting elbos...")
    elbo_lsts = []
    for file in os.listdir(statistics_dir):
        if file.startswith("elbos_"):
            elbo_lst = np.load(os.path.join(statistics_dir, file))["elbos"]
            elbo_lsts.append(elbo_lst)
    np.savez_compressed(os.path.join(statistics_dir, "elbo.npz"), l=np.mean(elbo_lsts, axis=0))


def compute_schedule(opt):
    """ Compute schedule maximizing training ELBO.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Create data normalizer and its inverse
    workdir = opt.workdir

    framework = EDM(opt.ckp_path) 

    statistics_dir = os.path.join(
        workdir, "statistics", f"{opt.ckp_path}__{opt.n_timesteps}_{num_gpus}_{opt.n_batch}_{opt.batch_size}"
    )
    os.makedirs(statistics_dir, exist_ok=True)

    compute_elbos(framework, statistics_dir, opt.n_batch, opt.n_timesteps, opt.batch_size, device)

    # import torch.multiprocessing as mp

    # mp.set_start_method(method="spawn", force=True)
    # print("Spawning processes...")
    # processes_l = [
    #     mp.Process(
    #         target=compute_elbos,
    #         args=(
    #             framework,
    #             statistics_dir,
    #             opt.n_batch,
    #             opt.n_timesteps,
    #             opt.batch_size,
    #             num_gpus,
    #             device,
    #             i,
    #         ),
    #     )
    #     for i in range(num_gpus)
    # ]

    # [p.start() for p in processes_l]
    # [p.join() for p in processes_l]

    # collect_elbos(statistics_dir)



def main(argv):
    compute_schedule(FLAGS)


if __name__ == "__main__":
    app.run(main)
