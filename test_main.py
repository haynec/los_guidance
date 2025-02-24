import subprocess

def run_main_with_params():
    params_files = [
        'quadsim/params/cinema_vp.py',
        'quadsim/params/camera_view.py',
        'quadsim/params/conic_view.py',
        'quadsim/params/scp_iters.py',
        'quadsim/params/constr_vio.py',
        'quadsim/params/control_losses.py',
        'quadsim/params/state.py',
        'quadsim/params/dr_vp.py',
        'quadsim/params/dr_vp_nodal.py',
        'quadsim/params/cinema_vp_nodal.py'
    ]

    plots = [
        'trajectory',
        'camera_view',
        'conic_view',
        'scp_iters',
        'constr_vio',
        'control_losses',
        'state',
        'dr_vp',
        'dr_vp_nodal',
        'cinema_vp_nodal'
    ]

    for params_file, plot in zip(params_files, plots):
        result = subprocess.run(['python', 'main.py', '--params-file', params_file, '--plot', plot], capture_output=True, text=True)
        assert result.returncode == 0, f"Process failed with return code {result.returncode}\n{result.stdout}\n{result.stderr}"