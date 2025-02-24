import subprocess

def test_main():
    result = subprocess.run(['python', 'main.py', '--params-file', 'quadsim/params/cinema_vp.py', '--params-file', 'trajectory camera_view conic_view scp_iters constr_vio control losses state'], capture_output=True, text=True)
    assert result.returncode == 0, f"Process failed with return code {result.returncode}\n{result.stdout}\n{result.stderr}"