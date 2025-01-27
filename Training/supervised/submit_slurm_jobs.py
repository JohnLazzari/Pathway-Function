import subprocess
import argparse
import os
import json

DEFAULTS = {
    "lr": 0.001,
    "weight_decay": 1e-3,
    "seed": 123456,
    "dt": 10,
    "t_const": 100,
    "batch_first": True,
    "activation_name": "relu",
    "constrained": True,
    "epochs": 100000,
    "batch_size": 1,
    "save_iter": 100,
    "device": "cpu",
    "noise_level_inp": 0.01,
    "noise_level_act": 0.15,
    "model_save_path": "checkpoints/mRNN_thal_inp.pth",
    "model_config_path": "Training/configurations/mRNN_thal_inp.json"
}

def fill_defaults(config):
    
    if "lr" not in config:
        config["lr"] = DEFAULTS["lr"]
    if "weight_decay" not in config:
        config["weight_decay"] = DEFAULTS["weight_decay"]
    if "seed" not in config:
        config["seed"] = DEFAULTS["seed"]
    if "dt" not in config:
        config["dt"] = DEFAULTS["dt"]
    if "t_const" not in config:
        config["t_const"] = DEFAULTS["t_const"]
    if "batch_first" not in config:
        config["batch_first"] = DEFAULTS["batch_first"]
    if "activation_name" not in config:
        config["activation_name"] = DEFAULTS["activation_name"]
    if "constrained" not in config:
        config["constrained"] = DEFAULTS["constrained"]
    if "epochs" not in config:
        config["epochs"] = DEFAULTS["epochs"]
    if "batch_first" not in config:
        config["batch_first"] = DEFAULTS["batch_first"]
    if "save_iter" not in config:
        config["save_iter"] = DEFAULTS["save_iter"]
    if "device" not in config:
        config["device"] = DEFAULTS["device"]
    if "noise_level_inp" not in config:
        config["noise_level_inp"] = DEFAULTS["noise_level_inp"]
    if "noise_level_act" not in config:
        config["noise_level_act"] = DEFAULTS["noise_level_act"]
    if "model_save_path" not in config:
        config["model_save_path"] = DEFAULTS["model_save_path"]
    if "model_config_path" not in config:
        config["model_config_path"] = DEFAULTS["model_config_path"]
        
    return config

def submit_job(job_name, params):
    """
    Submit a job to the Slurm cluster using `sbatch` command.
    """
    # Construct the parameter string for the sbatch script
    param_str = " ".join([f"--{key} {value}" for key, value in params.items()])

    slurm_script = f"""#!/bin/bash
                    #SBATCH --job-name={job_name}
                    #SBATCH --output={job_name}_%j.out
                    #SBATCH --error={job_name}_%j.err
                    #SBATCH --time=01:00:00
                    #SBATCH --mem=4GB

                    # Load modules (if any)
                    # module load conda

                    # Running your job with specified parameters
                    python motornet_optimization.py {param_str}
                    """

    # Write the Slurm batch script to a temporary file
    slurm_filename = f"{job_name}_slurm_script.sh"
    with open(slurm_filename, 'w') as f:
        f.write(slurm_script)

    # Submit the job using sbatch
    try:
        print(f"Submitting job {job_name} with parameters {param_str}")
        print("\n")
        result = subprocess.run(['sbatch', slurm_filename], capture_output=True, text=True, check=True)
        print(f"Job submitted successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
    finally:
        # Clean up the temporary slurm script file
        os.remove(slurm_filename)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)  
    args = parser.parse_args()

    # Load and process configuration
    with open(args.filename, 'r') as f:
        config = json.load(f)

    for job_name in config:
        job_spec = fill_defaults(config[job_name])
        # Submit the job
        submit_job(job_name, job_spec)

if __name__ == "__main__":
    main()