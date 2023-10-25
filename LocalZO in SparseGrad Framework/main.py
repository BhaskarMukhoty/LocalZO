import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nmnist', help='Dataset options: nmnist, fmnist, SHD ')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--nb_trials', type=int, default=1, help='Number of trials')
parser.add_argument('--nb_hidden', type=int, nargs='+', default=[200], help='Number hidden neurons in layers 1 and 2')
parser.add_argument('--prefix', type=str, default='RUN', help='Name of the experiment')
parser.add_argument('--nb_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--time_freq', type=int, default=20, help='How often we time')
parser.add_argument('--surrogate', type=str, default='normal', help='Surrogate options: normal, laplace, uniform, sigmoid, fsigmoid')

OPTIONS = parser.parse_args()

if __name__ == "__main__":

    # Run train and time
    from train_and_time import run
    dataset = OPTIONS.dataset
    hidden_list = OPTIONS.nb_hidden
    nb_trials = OPTIONS.nb_trials
    prs = {
        'PREFIX': OPTIONS.prefix,
        'nb_epochs': OPTIONS.nb_epochs,
        'seed': OPTIONS.seed,
        'time_freq': OPTIONS.time_freq,
        'surrogate': OPTIONS.surrogate,
    }
    run(dataset, hidden_list, nb_trials, prs=prs)
