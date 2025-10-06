
% =================================================================================
% =================================================================================
% Step 1.1  :  we unzip .gz files into the 'data' folder
% =================================================================================
% =================================================================================



% ==========================================
% Unzip .gz files into the 'data' folder
% ==========================================

% Define data folder (subfolder of current script location)
data_dir = fullfile(pwd, 'data');

% Ensure the 'data' folder exists
if ~exist(data_dir, 'dir')
    error('Data folder not found: %s', data_dir);
end

% List of gzipped files to extract
gz_files = {
    'all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle.csv.gz'
    'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv.gz'
    'all_ctgxyz_99genes_fillavg_fillzero_t012345_noshuffle.csv.gz'   % added file
};

% Loop through each file and extract
for i = 1:numel(gz_files)
    gz_path = fullfile(data_dir, gz_files{i});

    if ~isfile(gz_path)
        warning('⚠️ File not found: %s', gz_path);
        continue;
    end

    % Extract .csv to same folder
    fprintf('Extracting %s → %s\n', gz_path, data_dir);
    gunzip(gz_path, data_dir);
end

disp('✅ All .gz files extracted into the data folder.');





%%
% =================================================================================
% =================================================================================
% Step 1.2  :  attach g(t+1) for training VAE for each g(t)
% =================================================================================
% =================================================================================


% --- Base paths ---
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir) || contains(base_dir, 'Editor_')
    base_dir = pwd;  % fallback if run from MATLAB Editor temp folder
end
data_dir = fullfile(base_dir, 'data');

% --- Step 1: Import CSV ---
fname = fullfile(data_dir, 'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv');
disp('Reading CSV...');
T = readmatrix(fname);   % numeric matrix
[nrows, ncols] = size(T);
fprintf('Loaded %d rows × %d cols.\n', nrows, ncols);

% Expect ncols = 104 (cols 1=cellID, 2=time, 3–101=99 genes, 102–104=xyz)
if ncols ~= 104
    error('Expected 104 columns, but got %d.', ncols);
end

% --- Step 2: Prepare output matrix ---
Out = [T, nan(nrows, 99)];   % add 99 new columns (105–203)

% --- Step 3: Build lookup for fast access by (cellID,time) ---
cellIDs = T(:,1);
times   = T(:,2);

% Create containers.Map: key = cellID*10 + time (unique for each row)
keys = cellIDs * 10 + times;
[uniqueKeys, ia] = unique(keys, 'stable');
values = num2cell(ia);
keyMap = containers.Map(num2cell(uniqueKeys), values);

% --- Step 4: Fill shifted (t+1) gene values ---
fprintf('Filling shifted gene values...\n');
for k = 1:nrows
    cid = cellIDs(k);
    t   = times(k);
    if t < 5
        nextKey = cid * 10 + (t + 1);
        if isKey(keyMap, nextKey)
            nextRow = keyMap(nextKey);
            Out(k,105:203) = T(nextRow,3:101);
        end
    end
end

% --- Step 5: Replace NaN with 0 ---
Out(isnan(Out)) = 0;

% --- Step 6: Save result ---
out_fn = fullfile(data_dir, 'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv');
disp(['Saving to ', out_fn, ' ...']);
writematrix(Out, out_fn);


%%
% =================================================================================
% =================================================================================
% Step 1.3  :  shuffle the fillavg pre-imputation dataset
% =================================================================================
% =================================================================================


% --- Base paths ---
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir) || contains(base_dir, 'Editor_')
    base_dir = pwd;  % fallback if run from MATLAB Editor temp folder
end
data_dir = fullfile(base_dir, 'data');

% --- Step 1: Load fillrand (shuffled reference order) ---
fname_rand = fullfile(data_dir, 'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv');
disp('Reading shuffled fillrand file...');
T_ref = readmatrix(fname_rand);
fprintf('Loaded %d rows × %d cols from fillrand.\n', size(T_ref,1), size(T_ref,2));

% --- Step 2: Load fillavg (to be reordered) ---
fname_avg = fullfile(data_dir, 'all_ctgxyz_99genes_fillavg_fillzero_t012345_noshuffle.csv');
disp('Reading unshuffled fillavg file...');
T_avg = readmatrix(fname_avg);
fprintf('Loaded %d rows × %d cols from fillavg.\n', size(T_avg,1), size(T_avg,2));

% --- Step 3: Check basic consistency ---
if size(T_ref,2) ~= size(T_avg,2)
    error('Column mismatch between fillrand (%d) and fillavg (%d).', size(T_ref,2), size(T_avg,2));
end

% --- Step 4: Create lookup map for B (fillavg) by (cellID,time) ---
cellIDs_B = T_avg(:,1);
times_B   = T_avg(:,2);
keys_B = cellIDs_B * 10 + times_B;   % unique key = cellID*10 + time
[uniqueKeys_B, ia_B] = unique(keys_B, 'stable');
values_B = num2cell(ia_B);
map_B = containers.Map(num2cell(uniqueKeys_B), values_B);

% --- Step 5: Reorder B according to A ---
fprintf('Reordering fillavg to match shuffled fillrand order...\n');
nrows = size(T_ref,1);
Out = nan(size(T_ref));   % preallocate
cellIDs_A = T_ref(:,1);
times_A   = T_ref(:,2);
for k = 1:nrows
    keyA = cellIDs_A(k)*10 + times_A(k);
    if isKey(map_B, keyA)
        Out(k,:) = T_avg(map_B(keyA), :);
    else
        warning('No match found in fillavg for cellID=%d, time=%g', cellIDs_A(k), times_A(k));
    end
end

% --- Step 6: Sanity check ---
if any(isnan(Out(:,1)))
    warning('⚠️ Some rows did not find matches — check dataset consistency.');
else
    disp('✅ All rows successfully matched.');
end

% --- Step 7: Save reordered fillavg file ---
out_fn = fullfile(data_dir, 'all_ctgxyz_99genes_fillavg_fillzero_t012345_90percent_and_shuffled.csv');
disp(['Saving reordered fillavg to ', out_fn, ' ...']);
writematrix(Out, out_fn);
disp('✅ Done: fillavg now shuffled to match fillrand order.');




%%
% =================================================================================
% =================================================================================
% Step 1.4  :  add g(t+1) to the fillavg pre-imputation dataset
% =================================================================================
% =================================================================================


% --- Base paths ---
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir) || contains(base_dir, 'Editor_')
    base_dir = pwd;  % fallback if run from MATLAB Editor temp folder
end
data_dir = fullfile(base_dir, 'data');

% --- Step 1: Import shuffled fillavg CSV ---
fname = fullfile(data_dir, 'all_ctgxyz_99genes_fillavg_fillzero_t012345_90percent_and_shuffled.csv');
disp('Reading shuffled fillavg file...');
T = readmatrix(fname);   % numeric matrix
[nrows, ncols] = size(T);
fprintf('Loaded %d rows × %d cols.\n', nrows, ncols);

% Expect 104 columns: 1=cellID, 2=time, 3–101=99 genes, 102–104=xyz
if ncols ~= 104
    error('Expected 104 columns, but got %d.', ncols);
end

% --- Step 2: Prepare output matrix ---
Out = [T, nan(nrows, 99)];   % add 99 new columns (105–203)

% --- Step 3: Build lookup for fast access by (cellID,time) ---
cellIDs = T(:,1);
times   = T(:,2);
keys = cellIDs * 10 + times;
[uniqueKeys, ia] = unique(keys, 'stable');
values = num2cell(ia);
keyMap = containers.Map(num2cell(uniqueKeys), values);

% --- Step 4: Fill shifted (t+1) gene values ---
fprintf('Filling shifted gene values...\n');
for k = 1:nrows
    cid = cellIDs(k);
    t   = times(k);
    if t < 5
        nextKey = cid * 10 + (t + 1);
        if isKey(keyMap, nextKey)
            nextRow = keyMap(nextKey);
            Out(k,105:203) = T(nextRow,3:101);
        end
    end
end

% --- Step 5: Replace NaN with 0 ---
Out(isnan(Out)) = 0;

% --- Step 6: Save result ---
out_fn = fullfile(data_dir, 'all_ctgxyz_99genes_fillavg_fillzero_t012345_90percent_and_shuffled_tellnextg.csv');
disp(['Saving to ', out_fn, ' ...']);
writematrix(Out, out_fn);
disp('✅ Done: tellnextg file created for shuffled fillavg.');



%%
% =================================================================================
% =================================================================================
% Step 1.5  : Make fillzero_fillzero pre-imputation dataset
% =================================================================================
% =================================================================================

% --- Base paths ---
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir) || contains(base_dir, 'Editor_')
    base_dir = pwd;  % fallback for MATLAB Editor temp folder
end
data_dir = fullfile(base_dir, 'data');

% --- Input files to process ---
csv_files = {
    'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv'
    'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv'
    'all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle.csv'
};

% --- Process each file ---
for i = 1:numel(csv_files)
    in_path = fullfile(data_dir, csv_files{i});

    if ~isfile(in_path)
        warning('⚠️ File not found: %s', in_path);
        continue;
    end

    fprintf('\nProcessing %s\n', csv_files{i});

    % Read file
    T = readmatrix(in_path);
    fprintf('Loaded %d rows × %d cols.\n', size(T,1), size(T,2));

    % Replace negative values with 0
    T(T < 0) = 0;

    % Save as new file (insert "_fillzero_fillzero" before ".csv")
    [~, name, ext] = fileparts(in_path);
    out_path = fullfile(data_dir, [name '_fillzero_fillzero' ext]);

    fprintf('Saving cleaned file → %s\n', out_path);
    writematrix(T, out_path);
end

disp('✅ All fillzero_fillzero datasets created successfully.');
