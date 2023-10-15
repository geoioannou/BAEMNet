from experiments import run_exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '--names-list', nargs='+', default=["adult"],
                    help='Give the names of the datasets (e.g. --data adult german compas heloc)')
parser.add_argument('--name', default="shap_losses", 
                    help='Give the name of the results file')
parser.add_argument('--dev', default="cpu",
                    help='Give the name of the device')

args = parser.parse_args()


for dataset in args.datasets:
    run_exp(dset=dataset, dev=args.dev, name=args.name)