#!/bin/bash

#SBATCH --output=/projects/illinois/ahs/kch/nakhan2/Log_Files/Source_Localisation/Output/output_%x_%j.log  # Output log
#SBATCH --error=/projects/illinois/ahs/kch/nakhan2/Log_Files/Source_Localisation/Errors/error_%x_%j.log   # Error log
#SBATCH --time=2:00:00                # Max runtime (hh:mm:ss)
#SBATCH --cpus-per-task=1              # Number of CPUs
#SBATCH --mem=64G                      # Memory
#SBATCH --account=nakhan2-ic            # Group account
#SBATCH --partition=IllinoisComputes    # Compute partition
#SBATCH --nodes=1                      # Number of nodes


# Activate virtual environment
#change virtual environment if error occurs 
#source /projects/illinois/ahs/kch/nakhan2/venv/bin/activate
source /projects/illinois/ahs/kch/nakhan2/venv/bin/activate

# Run the Python script with subject_id as an argument
#for participants all in one
python /projects/illinois/ahs/kch/nakhan2/scripts/Step_2_Source_Localisation/plot_stc.py
# Deactivate virtual environment
deactivate
