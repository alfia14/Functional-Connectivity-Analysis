% Start EEGLAB without GUI
eeglab('nogui');

% Specify the folder containing the .cdt files
input_folder = '/Users/alfiaparvez/Documents/NURISH_N-back/N_Back'; % Folder containing your .cdt files
output_folder = '/Users/alfiaparvez/Documents/NURISH_N-back/EDF_Files'; % Folder to save the .edf files
event_folder = '/Users/alfiaparvez/Documents/NURISH_N-back/EventLists/'; % Folder to save the event files




% Create output folders if they don't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
if ~exist(event_folder, 'dir')
    mkdir(event_folder);
end



% List all .cdt files in the folder
cdt_files = dir(fullfile(input_folder, '*.cdt'));

participant_ids = {}; % Initialize an empty cell array for participant IDs
cond_list = {'WFC', 'WFC2'};   % List both FC1 and FC2 for processing
ncond = length(cond_list);

% Extract the unique participant IDs (e.g., ST101, ST102, etc.)
for s = 101:150          %range of particpant's ids
    studlist = 'NU';
    subject_base = sprintf('%s%03d', studlist, s); % Ensure leading zeros for participant ID (e.g., 'ST101')
    participant_ids{end+1} = subject_base; % Append base participant ID
end

fprintf("Flag2\n");

% Print participant IDs to verify them
disp('Participant IDs to be processed:');
disp(participant_ids);

% Loop through each participant and merge their FC1 and FC2 files
for i = 1:length(participant_ids)
    participant_id_base = participant_ids{i};
    
    % Construct the FC1 and FC2 file names for this participant
    fc1_file = fullfile(input_folder, [participant_id_base 'WFC1.cdt']); % Full file name for FC1
    fc2_file = fullfile(input_folder, [participant_id_base 'WFC2.cdt']); % Full file name for FC2
    
    % Check if both FC1 and FC2 files exist
    if isfile(fc1_file) && isfile(fc2_file)
        % Display progress
        fprintf('Processing participant %s...\n', participant_id_base);
        
        % Load FC1 and FC2 .cdt files
        EEG_FC1 = loadcurry(fc1_file, 'CurryLocations', 'False'); % Load without using Curry location files
        EEG_FC2 = loadcurry(fc2_file, 'CurryLocations', 'False'); % Load without using Curry location files
        
        % Check if the EEG structures are valid
        EEG_FC1 = eeg_checkset(EEG_FC1);
        EEG_FC2 = eeg_checkset(EEG_FC2);
        
        % Merge the datasets (FC1 and FC2)
        EEG_merged = pop_mergeset(EEG_FC1, EEG_FC2, 0); % The '0' flag indicates no GUI
        
        % Construct the output .edf file name
        edf_file_path = fullfile(output_folder, [participant_id_base '.edf']);    
        event_file_path = fullfile(event_folder, [participant_id_base '_events.txt']);
        
        % Save the merged data in EDF format
        pop_writeeeg(EEG_merged, edf_file_path, 'TYPE', 'EDF');
        
        % Extract event list and save to a .txt file
        EEG_merged = pop_creabasiceventlist(EEG_merged, 'AlphanumericCleaning', 'on', 'BoundaryNumeric', {-99}, 'BoundaryString', {'boundary'}, 'EventList', [event_folder participant_id_base '_EL.txt']);
        EEG_merged = pop_saveset(EEG_merged, event_file_path );
       
        
        % Notify conversion is complete for this participant
        fprintf('Successfully converted %s to %s and saved events.\n', participant_id_base, edf_file_path);
        
    else
        % Warn about missing files
        if ~isfile(fc1_file)
            fprintf('Missing FC1 file for participant %s. Skipping...\n',fc1_file);
        end
        if ~isfile(fc2_file)
            fprintf('Missing FC2 file for participant %s. Skipping...\n', fc2_file);
        end
    end
end

% Final message when done
fprintf('All participant files have been processed successfully.\n');
