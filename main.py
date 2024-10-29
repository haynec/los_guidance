from quadsim.ptr import PTR_main
from quadsim.plotting import plot_camera_view_ral, plot_conic_view_ral, plot_main_ral_dr, plot_main_ral_cine, plot_camera_animation, plot_animation, plot_scp_animation, plot_constraint_violation, plot_control, plot_state, plot_losses, plot_conic_view_animation, plot_camera_view
from quadsim.config import Config

import pickle
import argparse
import importlib.util
import warnings
warnings.filterwarnings("ignore")

###############################
# Author: Chris Hayner
# Autonomous Controls Laboratory
################################

def load_params_from_file(file_path):
    spec = importlib.util.spec_from_file_location("params", file_path)
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)
    return params_module.params

def run_simulation(params):
    print('Starting Quadrotor Simulator')

    params = Config.from_config(params, savedir="results/")
    results, _, _, _, _, _ = PTR_main(params) 

    # Save results
    with open('results/results.pickle', 'wb') as f:
        pickle.dump(results, f) 

    # Load results
    with open('results/results.pickle', 'rb') as f:
        results = pickle.load(f)
    
    return results

def plot_results(results, params, plots):
    if 'trajectory' in plots:
        plot_animation(results, params)
    if 'camera_view' in plots:
        plot_camera_animation(results, params)
    if 'conic_view' in plots:
        plot_conic_view_animation(results, params)
    if 'scp_iters' in plots:
        plot_scp_animation(results, None, params)
    if 'constr_vio' in plots:
        plot_constraint_violation(results, params)
    if 'control' in plots:
        plot_control(results, params)
    if 'losses' in plots:
        plot_losses(results, params)
    if 'state' in plots:
        plot_state(results, params)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Simulator with specified parameters.")
    parser.add_argument('--params-file', type=str, required=True, help="Path to the params file")
    parser.add_argument('--plots', type=str, nargs='*', choices=['trajectory', 'camera_view', 'conic_view', 'scp_iters', 'constr_vio', 'control', 'losses', 'state'], help="List of plots to generate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = load_params_from_file(args.params_file)
    results = run_simulation(params)

    if args.plots:
        plot_results(results, params, args.plots)