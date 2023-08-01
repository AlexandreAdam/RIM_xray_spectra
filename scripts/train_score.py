import torch
from score_models import ScoreModel, NCSNpp
from torch.func import vmap
import os, json
import pickle
import numpy as np
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    true_spectra_data = pickle.load(open(os.path.join(args.data_path, 'true_original.pkl'), 'rb'))
    min_ = 35
    max_ = 175

    # Read the spectra data
    X = np.stack([data[1][1][min_:max_] for data in true_spectra_data.items()])

    with open(args.hyperparameters, "r") as f:
        hyperparameters = json.load(f)

    # Make a dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, index):
            x = torch.tensor(X[index]).float().to(DEVICE)
            return x.view(1, -1) # add the channel

    dataset = Dataset(len(X))
    checkpoints_directory = args.checkpoints_directory
    model = NCSNpp(**hyperparameters).to(DEVICE)
    score_model = ScoreModel(model, **hyperparameters)

    score_model.fit(
        dataset, 
        checkpoints_directory=checkpoints_directory, 
        logname=args.logname,
        logname_prefix=args.logname_prefix,
        logdir=args.logdir,
        epochs=args.epochs,
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        checkpoints=args.checkpoints,
        models_to_keep=args.models_to_keep,
        ema_decay=args.ema_decay,
        max_time=args.max_time
    )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoints_directory", default=None)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--logname", default=None)
    parser.add_argument("--logname_prefixe", default="score")
    parser.add_argument("--hyperparameters", required=True, help="Path to hyperparameters json file")
    
    # optim parameters
    parser.add_argument("--epochs", default=20000, type=int, help="Number of epochs to run this")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay. If decay=0, EMA is turned off.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of examples to feed in")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--checkpoints", default=5, type=int, help="Save model every 'checkpoints' epochs")
    parser.add_argument("--models_to_keep", default=3, type=int)
    parser.add_argument("--max_time", default=np.inf, type=float, help="Maximum of time allowed for training in hours")
    
    args = parser.parse_args()
    main(args)

