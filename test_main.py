import subprocess
from unittest.mock import patch

def run_main_with_params(params):
    result = subprocess.run(['python', 'main.py', '--params-file', 'quadsim/params/cinema_vp.py', '--params', params], capture_output=True, text=True)
    assert result.returncode == 0, f"Process failed with return code {result.returncode}\n{result.stdout}\n{result.stderr}"

@patch('main.plot_func')  # Mock the plotting function
def test_trajectory(mock_plot_func):
    run_main_with_params('trajectory')
    mock_plot_func.assert_called_once()

# def test_camera_view():
#     run_main_with_params('camera_view')

# def test_conic_view():
#     run_main_with_params('conic_view')

# def test_scp_iters():
#     run_main_with_params('scp_iters')

# def test_constr_vio():
#     run_main_with_params('constr_vio')

# def test_control_losses():
#     run_main_with_params('control_losses')

# def test_state():
#     run_main_with_params('state')