import torch
from rim import RIM, Hourglass
from score_models import ScoreModel
from torch.func import vmap
import pickle
import numpy as np
import os, json
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    responses_data = pickle.load(open(os.path.join(args.data_path, 'rmfs_original.pkl'), 'rb'))
    true_spectra_data = pickle.load(open(os.path.join(args.data_path, 'true_original.pkl'), 'rb'))
    min_ = 35
    max_ = 175
    
    if args.prior_model is not None:
        prior = ScoreModel(checkpoints_directory=args.prior_model, dimensions=1)
        temperature = lambda batch_size: torch.ones(batch_size).to(DEVICE) * args.prior_temperature # at which temperature to evaluate the prior
        def prior_score(x):
            B, *D = x.shape
            return prior.score(t=temperature(B), x=x)
    else:
        prior_score = lambda x: 0.

    # Read in A, and x
    X = np.stack([data[1][1][min_:max_] for data in true_spectra_data.items()])
    if args.transpose_response:
        print("Using transposed response matrix")
        A_dataset = np.stack([responses_data[val][min_:max_,min_:max_].T for val in responses_data]) #
    else:
        A_dataset = np.stack([responses_data[val][min_:max_,min_:max_] for val in responses_data]) #
    
    snr_max = args.snr_max
    snr_min = args.snr_min
    m = A_dataset[0].shape[0] # model space
    n = A_dataset[0].shape[1] # observation space

    @vmap
    def likelihood_score_fn(x, y, A, sigma_n): # make sure to respect this signature (x, y, *args)
        y_pred = A @ x.squeeze() # remove channel dimension of x.
        score = - (y - y_pred).T @ (-A) / sigma_n**2
        return score.unsqueeze(0) # give score back its channel dimensions
    
    def score_fn(x, y, A, sigma_n):
        return likelihood_score_fn(x, y, A, sigma_n) + prior_score(x)

    # Make a dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, index):
            k1 = np.random.randint(len(X))
            k2 = np.random.randint(len(A_dataset))
            x = torch.tensor(X[k1]).float().to(DEVICE)
            A = torch.tensor(A_dataset[k2]).float().to(DEVICE)
            y = A @ x # noiseless observation
            # Uniform distribution in SNR
            if args.noise_distribution == "uniform":
                snr = torch.rand([]).to(DEVICE) * (snr_max - snr_min) + snr_min
                sigma_n = y.max() / snr
            elif args.noise_distribution == "log_uniform":
                log_snr = torch.rand([]).to(DEVICE) * (np.log10(snr_max) - np.log10(snr_min)) + np.log10(snr_min)
                snr = 10**log_snr
                sigma_n = y.max() / snr
                
            z = torch.randn([n]).to(DEVICE) * sigma_n
            y = y + z # noisy observation
            # For dataset, it's the traditional ml signature (inputs, labels, *args)
            return y, x.view(1, -1), A, sigma_n # add channel dimension to x

    with open(args.hyperparameters, "r") as f:
        hyperparameters = json.load(f)
        
    dataset = Dataset(len(X))
    checkpoints_directory = args.checkpoints_directory
    model = Hourglass(**hyperparameters).to(DEVICE)
    rim = RIM(dimensions=[m], model=model, score_fn=score_fn, device=DEVICE, **hyperparameters)
    lr_schedule = lambda optim: torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_schedule_step_size, gamma=args.lr_schedule_gamma)
    
    rim.fit(
        dataset, 
        checkpoints_directory=checkpoints_directory, 
        logname=args.logname,
        logdir=args.logdir,
        logname_prefix=args.logname_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        scheduler=lr_schedule,
        checkpoints=args.checkpoints,
        models_to_keep=args.models_to_keep,
        clip=args.clip,
        ema_decay=args.ema_decay,
        max_time=args.max_time
    )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prior_model", default=None, help="Path to prior score model. Default is None, not using a score model for the prior.")
    parser.add_argument("--prior_temperature", default=0, type=float, help="Temperature at which to evaluate prior model. Default is 0.")
    parser.add_argument("--transpose_response", action="store_true", help="Transpose the response matrix from the data")
    parser.add_argument("--checkpoints_directory", default=None)
    parser.add_argument("--logname", default=None)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--logname_prefix", default="rim")
    parser.add_argument("--snr_max", default=100, type=float)
    parser.add_argument("--snr_min", default=5, type=float)
    parser.add_argument("--noise_distribution", default="log_uniform", help="Supported distribution are uniform and log_uniform")
    parser.add_argument("--hyperparameters", required=True, help="Path to hyperparameters json file")
    
    # optim parameters
    parser.add_argument("--epochs", default=20000, type=int, help="Number of epochs to run this")
    parser.add_argument("--clip", default=1, type=float, help="Gradient clipping")
    parser.add_argument("--ema_decay", default=0, type=float, help="EMA decay. If decay=0, EMA is turned off.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of examples to feed in")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--checkpoints", default=1, type=int, help="Save model every 'checkpoints' epochs")
    parser.add_argument("--models_to_keep", default=3, type=int)
    parser.add_argument("--max_time", default=np.inf, type=float, help="Maximum of time allowed for training in hours")
    parser.add_argument("--lr_schedule_step_size", default=500, type=int, help="Number of epochs until we dcrease the learning rate")
    parser.add_argument("--lr_schedule_gamma", default=0.8, type=float, help="Factor by which to reduce learning rate after step_size")
    
    args = parser.parse_args()
    main(args)
