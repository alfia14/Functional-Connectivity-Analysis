#!/bin/bash

#SBATCH --output=/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Source_Localisation/Output/output_%x_%j.log  # Output log
#SBATCH --error=/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Source_Localisation/Errors/error_%x_%j.log   # Error log
#SBATCH --time=24:00:00                # Max runtime (hh:mm:ss)
#SBATCH --cpus-per-task=1              # Number of CPUs
#SBATCH --mem=64G                      # Memory
#SBATCH --account=nakhan2-ic            # Group account
#SBATCH --partition=IllinoisComputes    # Compute partition
#SBATCH --nodes=1                      # Number of nodes

# Check if subject_id is provided
if [ -z "$1" ]; then

    echo "Error: No subject_id provided."
    echo "Usage: sbatch eeg_analysis.sh <subject_id>"
    exit 1
fi

# Assign the argument to subject_id
subject_id=$1

# Activate virtual environment
#change source according to your virtual environment name
#source /projects/illinois/ahs/kch/nakhan2/Shreya/bin/activate
source /projects/illinois/ahs/kch/nakhan2/Shreya/bin/activate

# Run the Python script with subject_id as an argument
#keep changing the path of the python scripts that you want to run
#python /projects/illinois/ahs/kch/nakhan2/scripts/Step_2_Source_Localisation/SR_congruent.py --subject_id "$subject_id"
#python /projects/illinois/ahs/kch/nakhan2/scripts/Step_2_Source_Localisation/SR_Incongruent.py --subject_id "$subject_id"
#python /projects/illinois/ahs/kch/nakhan2/scripts/Step_2_Source_Localisation/Source_Parcel.py --subject_id "$subject_id"
#python /projects/illinois/ahs/kch/nakhan2/scripts/Step_3_Brain_States/Orthogonalization.py --subject_id "$subject_id"
python /projects/illinois/ahs/kch/nakhan2/scripts/Step_3_Brain_States/Optimal_States.py --subject_id "$subject_id"
# Deactivate virtual environment
deactivate