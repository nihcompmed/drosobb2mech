function PlotFigure3()
% One figure, 2x3 layout:
%   Row 1 = batch 250915 (recon+KL+sparsity only; no jacdec/directional)
%   Row 2 = batch 250916 (full schedule with jacdec/directional)
%   Cols  = MAE_99, MSE_99, meanR2_99
% Plots only suffix == 'valid'.
% Skips ZERO runs at latent dim = 8 (outlier).
% Uses large fonts, thick lines, big markers, box on, and R^2 top limit = 1.

% ---------- Base path ----------
base_dir = fileparts(mfilename('fullpath'));  % path of this script (plotfigures/)
data_dir = fullfile(base_dir, '..', 'data');  % go one level up, into /data

% ---------- Files ----------
files915.zero = fullfile(data_dir, 'summary_recon_metrics_zero.csv');
files915.avg  = fullfile(data_dir, 'summary_recon_metrics_avg.csv');
files915.rand = fullfile(data_dir, 'summary_recon_metrics_rand.csv');

files916.zero = fullfile(data_dir, 'summary_recon_metrics_zero_250916.csv');
files916.avg  = fullfile(data_dir, 'summary_recon_metrics_avg_250916.csv');
files916.rand = fullfile(data_dir, 'summary_recon_metrics_rand_250916.csv');


% ---------- Read ----------
T915.zero = local_read_table(files915.zero);
T915.avg  = local_read_table(files915.avg);
T915.rand = local_read_table(files915.rand);

T916.zero = local_read_table(files916.zero);
T916.avg  = local_read_table(files916.avg);
T916.rand = local_read_table(files916.rand);

% ---------- jobid → latent dim map ----------
ldims = [2 3 4 5 6 7 8 9 10 12 14 16];

pairs = [ ...
  (250915001:250915012).' , ldims.' ;  % zero (915)
  (250915013:250915024).' , ldims.' ;  % avg
  (250915025:250915036).' , ldims.' ;  % rand
  (250916001:250916012).' , ldims.' ;  % zero (916)
  (250916013:250916024).' , ldims.' ;  % avg
  (250916025:250916036).' , ldims.' ]; % rand

jobid_to_ldim = containers.Map(pairs(:,1), pairs(:,2));

% ---------- Attach ldim + batch labels ----------
for nm = ["zero","avg","rand"]
    T915.(nm).ldim = local_lookup_ldim(T915.(nm).jobid, jobid_to_ldim);
    T916.(nm).ldim = local_lookup_ldim(T916.(nm).jobid, jobid_to_ldim);

    T915.(nm).batch = repmat("250915", height(T915.(nm)), 1);
    T916.(nm).batch = repmat("250916", height(T916.(nm)), 1);
end

% ---------- Keep ONLY 'valid' rows ----------
for nm = ["zero","avg","rand"]
    T915.(nm) = T915.(nm)(strcmp(string(T915.(nm).suffix), "valid") & ~isnan(T915.(nm).ldim), :);
    T916.(nm) = T916.(nm)(strcmp(string(T916.(nm).suffix), "valid") & ~isnan(T916.(nm).ldim), :);
end

% ---------- Outlier: ZERO @ ldim = 8 -> NaN (skip in plot) ----------
m99 = ["AE_mean_99","SE_mean_99","R2_mean_99"];
maskZero8 = T915.zero.ldim == 8 | T916.zero.ldim == 8;
for v = m99
    T915.zero.(v)(maskZero8) = NaN;
%     T916.zero.(v)(maskZero8) = NaN;
end

% ---------- Per-batch grouped means (by ldim) ----------
B915.zero = local_group_means_valid99(T915.zero, "250915");
B915.avg  = local_group_means_valid99(T915.avg,  "250915");
B915.rand = local_group_means_valid99(T915.rand, "250915");

B916.zero = local_group_means_valid99(T916.zero, "250916");
B916.avg  = local_group_means_valid99(T916.avg,  "250916");
B916.rand = local_group_means_valid99(T916.rand, "250916");


dims915 = local_union_dims_3(B915);
dims916 = local_union_dims_3(B916);

% ---------- Styling ----------
baseFS   = 22;
labelFS  = 24;
legendFS = 22;
LW       = 3.0;
MS       = 11;

% ---------- Plot ----------
f = figure('Color','w','Name','valid / 99 metrics (2x3)');
t = tiledlayout(2,3, 'Padding','compact', 'TileSpacing','compact');
set(f, 'DefaultAxesFontSize', baseFS);

mfields = {'AE_mean_99','SE_mean_99','R2_mean_99'};
ylabs   = {'MAE','MSE','mean R^2'};

% ---- Row 1: 250915 ----
for c = 1:3
    nexttile;
    hold on;

    [x0,y0] = local_align_curve(B915.zero, mfields{c}, dims915);
    [x1,y1] = local_align_curve(B915.avg,  mfields{c}, dims915);
    [x2,y2] = local_align_curve(B915.rand, mfields{c}, dims915);

    plot(x0, y0, '-o', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','zero');
    plot(x1, y1, '-s', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','avg');
    plot(x2, y2, '-^', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','rand');

    grid on; box on;
    ylabel(ylabs{c}, 'FontSize', labelFS);
    xticks(dims915);
     if c==2
        ytickformat('%.3f');
    else
        ytickformat('%.2f');
    end
    if strcmp(mfields{c}, 'R2_mean_99')
        ylim(local_r2_ylim_top1([y0;y1;y2]));
    end
end

% ---- Row 2: 250916 ----
ax = gobjects(1,3);
for c = 1:3
    ax(c) = nexttile;
    hold on;

    [x0,y0] = local_align_curve(B916.zero, mfields{c}, dims916);
    [x1,y1] = local_align_curve(B916.avg,  mfields{c}, dims916);
    [x2,y2] = local_align_curve(B916.rand, mfields{c}, dims916);

    plot(x0, y0, '-o', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','zero');
    plot(x1, y1, '-s', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','avg');
    plot(x2, y2, '-^', 'LineWidth',LW, 'MarkerSize',MS, 'DisplayName','rand');

    grid on; box on;
    ylabel(ylabs{c}, 'FontSize', labelFS);
    xlabel('Latent dim', 'FontSize', labelFS);
    xticks(dims916);
    if c==2
        ytickformat('%.3f');
    else
        ytickformat('%.2f');
    end
    if strcmp(mfields{c}, 'R2_mean_99')
        ylim(local_r2_ylim_top1([y0;y1;y2]));
    end
end

lg = legend(ax(end), {'zero','mean','random'});
lg.Orientation = 'horizontal';
lg.Location = 'southoutside';
lg.Box = 'off';
lg.FontSize = legendFS;

end % ===== main =====


% ===================== Helpers =====================

function T = local_read_table(fname)
    if ~isfile(fname), error('File not found: %s', fname); end
    T = readtable(fname);
    if iscell(T.jobid), T.jobid = string(T.jobid); end
    if isnumeric(T.jobid), T.jobid = string(T.jobid); end
    if ~isstring(T.suffix), T.suffix = string(T.suffix); end
    mcols = ["AE_mean_99","SE_mean_99","R2_mean_99","AE_mean_27","SE_mean_27","R2_mean_27"];
    for k = 1:numel(mcols)
        if ismember(mcols(k), T.Properties.VariableNames)
            T.(mcols(k)) = double(T.(mcols(k)));
        end
    end
end

function ldim = local_lookup_ldim(jobid_str, mp)
    jid = str2double(jobid_str);
    ldim = nan(size(jid));
    for i = 1:numel(jid)
        if ~isnan(jid(i)) && isKey(mp, jid(i)), ldim(i) = mp(jid(i)); end
    end
end

function batch = local_batch_label(jobid_str)
    jid = str2double(jobid_str);
    prefix = floor(jid/1000);
    batch = strings(size(jid));
    batch(prefix==250911) = "250911";
    batch(prefix==250915) = "250915";
    batch(~ismember(prefix,[250911,250915])) = "unknown";
end

function G = local_group_means_valid99(T, batchStr)
    G = table();
    if isempty(T), return; end
    Ts = T(string(T.batch)==batchStr & ~isnan(T.ldim), :);
    if isempty(Ts), return; end
    vars = ["AE_mean_99","SE_mean_99","R2_mean_99"];
    S = groupsummary(Ts, "ldim", "mean", vars);
    if isempty(S), return; end
    G.ldim = S.ldim;
    for v = vars, G.(v) = S.("mean_"+v); end
    [G.ldim, ord] = sort(G.ldim);
    for v = vars, G.(v) = G.(v)(ord); end
end

function dims = local_union_dims_3(B)
    dims = [];
    if ~isempty(B.zero), dims = union(dims, B.zero.ldim); end
    if ~isempty(B.avg),  dims = union(dims, B.avg.ldim);  end
    if ~isempty(B.rand), dims = union(dims, B.rand.ldim); end
end

function [x,y] = local_align_curve(Gtbl, mfield, xdims)
    x = xdims; y = nan(size(xdims));
    if isempty(Gtbl) || ~ismember(mfield, Gtbl.Properties.VariableNames), return; end
    [tf,loc] = ismember(Gtbl.ldim, xdims);
    y(loc(tf)) = Gtbl.(mfield)(tf);
end

function R = local_r2_ylim_top1(vals)
    vals = vals(isfinite(vals));
    if isempty(vals)
        R = [0 1];
        return;
    end
    ymin = max(min(vals) - 0.02, 0);
    if ymin >= 1, ymin = 0.95; end
    R = [ymin 1];  % TOP RANGE FIXED AT 1
end