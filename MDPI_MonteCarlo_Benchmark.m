function MDPI_MonteCarlo_Benchmark
    % =====================================================================
    % DOCTORAL THESIS: MONTE CARLO STATISTICAL BENCHMARKING SUITE
    % Runs N=50 iterations per algorithm to evaluate stochastic robustness.
    % Generates Publication-Quality Boxplots and Statistical HTML Reports.
    % =====================================================================
    clc; close all; warning('off', 'all');
    
    disp('======================================================');
    disp(' INITIALIZING MONTE CARLO BENCHMARKING SUITE (N=50) ');
    disp('======================================================');
    
    %% 1. CITY SELECTION & BASE PARAMETERS
    Cities(1) = struct('Name','Ambato, ECU', 'Lat',-1.2417, 'Lon',-78.6197, 'Alt',2580);
    Cities(2) = struct('Name','Cuenca, ECU', 'Lat',-2.9001, 'Lon',-79.0059, 'Alt',2560);
    Cities(3) = struct('Name','Quito, ECU',  'Lat',-0.1807, 'Lon',-78.4678, 'Alt',2850);
    
    [indx, tf] = listdlg('PromptString', 'Select the Target City for Monte Carlo:', ...
                         'SelectionMode', 'single', 'ListString', {Cities.Name}, 'ListSize', [250 120]);
    if ~tf, disp('Operation cancelled by user.'); return; end
    City = Cities(indx);
    
    % Input Dialog for RF Parameters
    prompt = {'Spreading Factor (SF 7 to 12):', 'Gateway Antenna Height (meters):', 'Number of Iterations (N):'};
    dlgtitle = 'Monte Carlo Settings'; dims = [1 65]; definput = {'11', '6', '50'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    if isempty(answer), disp('Operation cancelled.'); return; end
    
    HW.SF = str2double(answer{1});
    HW.Height = str2double(answer{2});
    N_Runs = str2double(answer{3});
    
    Target_Cov = 0.98; 
    HW.Name = 'Milesight UG67 Outdoor'; HW.Freq = 915; HW.Ptx = 27; HW.Gain = 5; HW.PenetrationLoss = 0; 
    Sens_Map = containers.Map({7,8,9,10,11,12}, {-125, -128, -131, -134, -136, -137});
    HW.Sens = Sens_Map(HW.SF);
    Algo_Names = ["Greedy SCP", "K-Medoids 3D", "Standard SAGA", "Proposed 3D-M-SAGA"];
    
    % Dynamic Folder Creation
    city_clean = strrep(strrep(City.Name, ', ', '_'), ' ', ''); 
    fecha_hora = datestr(now, 'yyyymmdd_HHMMSS');
    folder_name = sprintf('MONTECARLO_%s_%s', city_clean, fecha_hora);
    mkdir(folder_name);
    
    % Prepare Master Results Table
    VarNames = {'Iteration', 'Algorithm', 'Gateways', 'Coverage_Pct', 'K1_Risk', 'K2_Plus_Resilience', 'Time_s'};
    VarTypes = {'double', 'string', 'double', 'double', 'double', 'double', 'double'};
    MC_Results = table('Size', [0, length(VarNames)], 'VariableTypes', VarTypes, 'VariableNames', VarNames);
    
    %% 2. BUILD DIGITAL TWIN & PRECOMPUTE RF MATRIX (Done once!)
    disp(' -> [PHASE 1] Building 3D Digital Twin and Computing RF Path-Loss...');
    T_Twin = Build_DigitalTwin(City, 0.020); 
    if isempty(T_Twin), return; end
    Pts_3D = [T_Twin.X_m, T_Twin.Y_m, T_Twin.Altitude_m];
    
    RSSI_Matrix = Precompute_RSSI(T_Twin, HW);
    BinM_3D = RSSI_Matrix >= HW.Sens;
    max_cov = sum(sum(BinM_3D, 2) > 0) / height(T_Twin);
    Actual_Target = min(Target_Cov, max_cov * 0.99);
    
    %% 3. MONTE CARLO EXECUTION LOOP
    disp(' ');
    fprintf(' -> [PHASE 2] Starting Monte Carlo Loop (%d Iterations)...\n', N_Runs);
    
    for iter = 1:N_Runs
        rng('shuffle'); % Ensure stochastic variance across iterations
        fprintf('    Running Iteration %d / %d...\n', iter, N_Runs);
        
        for a = 1:4
            tic;
            if a == 1, sol = Run_Greedy(BinM_3D, Actual_Target);
            elseif a == 2, sol = Run_KMedoids(Pts_3D, BinM_3D, Actual_Target);
            elseif a == 3, sol = Run_GA(BinM_3D, Actual_Target, false);
            elseif a == 4, sol = Run_GA(BinM_3D, Actual_Target, true);
            end
            t_exec = toc;
            
            % Extract KPIs
            idx = find(sol); n_gws = length(idx); N_nodes = size(BinM_3D, 1);
            if n_gws > 0
                mapa = sum(BinM_3D(:, idx), 2); 
                c_tot = round((sum(mapa >= 1)/N_nodes)*100, 2);
                k1 = round((sum(mapa == 1)/N_nodes)*100, 2); 
                k2_plus = round((sum(mapa >= 2)/N_nodes)*100, 2); % K>=2 Resilience
            else
                c_tot = 0; k1 = 0; k2_plus = 0;
            end
            
            row = {iter, Algo_Names(a), n_gws, c_tot, k1, k2_plus, round(t_exec, 3)};
            MC_Results = [MC_Results; row];
        end
    end
    
    %% 4. STATISTICAL ANALYSIS, PLOTTING & HTML EXPORT
    disp(' -> [PHASE 3] Generating Statistical Plots and HTML Report...');
    
    writetable(MC_Results, fullfile(folder_name, 'Raw_MonteCarlo_Data.csv'));
    
    Generate_Publication_Plots(MC_Results, Algo_Names, HW, N_Runs, folder_name);
    Generate_Statistical_HTML(City, HW, N_Runs, MC_Results, Algo_Names, folder_name);
    
    disp('======================================================');
    disp([' -> SUCCESS! All Monte Carlo assets saved in: ', folder_name]);
    disp('======================================================');
end

%% ========================================================================
%% PUBLICATION QUALITY PLOTS (BOXPLOTS)
%% ========================================================================
function Generate_Publication_Plots(Data, Algo_Names, HW, N, folder)
    % Prepare categorical data for ordered plotting
    Data.Algorithm = categorical(Data.Algorithm, Algo_Names);
    c_blue = [0 0.4470 0.7410];
    
    % PLOT 1: CAPEX (Number of Gateways)
    f1 = figure('Name', 'MonteCarlo_CAPEX', 'Position', [100, 100, 700, 500], 'Color', 'w');
    boxchart(Data.Algorithm, Data.Gateways, 'BoxFaceColor', c_blue, 'MarkerStyle', 'x');
    title(sprintf('Infrastructure CAPEX Variability (N=%d Runs, SF%d, H=%dm)', N, HW.SF, HW.Height), 'FontSize', 12);
    ylabel('Number of Gateways Deployed'); grid on;
    exportgraphics(f1, fullfile(folder, 'Fig_MC_Gateways.png'), 'Resolution', 600);
    savefig(f1, fullfile(folder, 'Fig_MC_Gateways.fig'));
    
    % PLOT 2: Network Resilience (K >= 2 Overlap)
    f2 = figure('Name', 'MonteCarlo_Resilience', 'Position', [150, 150, 700, 500], 'Color', 'w');
    boxchart(Data.Algorithm, Data.K2_Plus_Resilience, 'BoxFaceColor', [0.4660 0.6740 0.1880], 'MarkerStyle', 'x');
    title(sprintf('Network Resilience / Fault Tolerance (K \\geq 2)'), 'FontSize', 12);
    ylabel('Intersections with Overlapping Redundancy (%)'); grid on;
    exportgraphics(f2, fullfile(folder, 'Fig_MC_Resilience.png'), 'Resolution', 600);
    savefig(f2, fullfile(folder, 'Fig_MC_Resilience.fig'));
    
    % PLOT 3: Physical Coverage (%)
    f3 = figure('Name', 'MonteCarlo_Coverage', 'Position', [200, 200, 700, 500], 'Color', 'w');
    boxchart(Data.Algorithm, Data.Coverage_Pct, 'BoxFaceColor', [0.8500 0.3250 0.0980], 'MarkerStyle', 'x');
    title(sprintf('Physical Coverage Stability'), 'FontSize', 12);
    ylabel('Total Topographic Coverage (%)'); grid on;
    exportgraphics(f3, fullfile(folder, 'Fig_MC_Coverage.png'), 'Resolution', 600);
    savefig(f3, fullfile(folder, 'Fig_MC_Coverage.fig'));
end

%% ========================================================================
%% HTML STATISTICAL REPORT GENERATOR
%% ========================================================================
%% ========================================================================
%% HTML STATISTICAL REPORT GENERATOR (CORREGIDO)
%% ========================================================================
function Generate_Statistical_HTML(City, HW, N, Data, Algo_Names, folder)
    file = fullfile(folder, 'MonteCarlo_Statistical_Report.html'); fid = fopen(file, 'w', 'n', 'UTF-8');
    
    fprintf(fid, '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{font-family:Arial, sans-serif;margin:40px;color:#333;line-height:1.6;} table{border-collapse:collapse;width:100%%;margin-bottom:40px;} th,td{border:1px solid #ccc;padding:10px;text-align:center;} th{background-color:#003366;color:white;} .box{background-color:#f0f4f8;padding:20px;border-left:5px solid #003366;margin-bottom:20px;} h2{color:#003366;border-bottom:2px solid #ccc;padding-bottom:5px;}</style></head><body>\n');
    fprintf(fid, '<h1>Monte Carlo Statistical Benchmarking Report</h1>\n');
    fprintf(fid, '<div class="box"><p><strong>City:</strong> %s</p><p><strong>RF Configuration:</strong> SF%d | %d meters Height</p><p><strong>Independent Runs (N):</strong> %d</p></div>\n', City.Name, HW.SF, HW.Height, N);
    
    fprintf(fid, '<h2>Statistical Aggregation (\x03BC \x00B1 \x03C3)</h2>\n');
    fprintf(fid, '<p>This table presents the Mean (\x03BC) and Standard Deviation (\x03C3) across all iterations, demonstrating the stochastic stability of the algorithms.</p>\n');
    
    % LA LÍNEA CORREGIDA: Se usa doble porcentaje (%%) en Coverage para que MATLAB no lo confunda
    fprintf(fid, '<table><tr><th>Algorithm</th><th>Gateways (CAPEX)</th><th>Coverage (%%)</th><th>K=1 (Fragility)</th><th>K&ge;2 (Resilience)</th><th>Execution Time (s)</th></tr>\n');
    
    for i = 1:length(Algo_Names)
        SubD = Data(Data.Algorithm == Algo_Names(i), :);
        if isempty(SubD), continue; end
        
        m_gw = mean(SubD.Gateways); s_gw = std(SubD.Gateways);
        m_cv = mean(SubD.Coverage_Pct); s_cv = std(SubD.Coverage_Pct);
        m_k1 = mean(SubD.K1_Risk); s_k1 = std(SubD.K1_Risk);
        m_k2 = mean(SubD.K2_Plus_Resilience); s_k2 = std(SubD.K2_Plus_Resilience);
        m_t = mean(SubD.Time_s); s_t = std(SubD.Time_s);
        
        highlight = ''; if i == 4, highlight = 'background-color:#e3f2fd; font-weight:bold;'; end
        
        fprintf(fid, '<tr style="%s"><td>%s</td><td>%.2f &plusmn; %.2f</td><td>%.2f%% &plusmn; %.2f</td><td>%.2f%% &plusmn; %.2f</td><td>%.2f%% &plusmn; %.2f</td><td>%.3fs &plusmn; %.3f</td></tr>\n', ...
            highlight, Algo_Names(i), m_gw, s_gw, m_cv, s_cv, m_k1, s_k1, m_k2, s_k2, m_t, s_t);
    end
    
    fprintf(fid, '</table></body></html>\n'); fclose(fid);
end
%% ========================================================================
%% CORE FUNCTIONS (TWIN, RF, ALGORITHMS)
%% ========================================================================
function T = Build_DigitalTwin(City, dist)
    min_lat = City.Lat - dist; min_lon = City.Lon - dist; max_lat = City.Lat + dist; max_lon = City.Lon + dist;
    options = weboptions('Timeout', 120); 
    q_osm = sprintf('[out:json][timeout:120];(way["highway"~"^(primary|secondary|tertiary|residential)$"](%f,%f,%f,%f););out body;>;out skel qt;', min_lat, min_lon, max_lat, max_lon);
    try data = webread('https://lz4.overpass-api.de/api/interpreter', 'data', q_osm, options); catch, T=[]; return; end
    c_nodes = containers.Map('KeyType','uint64','ValueType','any'); c_names = containers.Map('KeyType','uint64','ValueType','any');
    for i=1:length(data.elements), e = data.elements(i); if iscell(data.elements), e=e{1}; end
        if strcmp(e.type,'node'), c_nodes(e.id) = [e.lat, e.lon];
        elseif strcmp(e.type,'way') && isfield(e, 'nodes')
            street = "Road"; if isfield(e, 'tags') && isfield(e.tags, 'name'), street = string(e.tags.name); end
            for k=1:length(e.nodes), nid = e.nodes(k); if iscell(e.nodes), nid=nid{1}; end
                if isKey(c_names, nid), c_names(nid) = [c_names(nid), street]; else, c_names(nid) = street; end
            end
        end
    end
    lat = []; lon = []; keys_n = c_names.keys;
    for k = 1:length(keys_n), nid = keys_n{k}; noms = unique(c_names(nid)); 
        if length(noms) >= 2 && isKey(c_nodes, nid), coords = c_nodes(nid); lat = [lat; coords(1)]; lon = [lon; coords(2)]; end
    end
    N = length(lat); Alt_m = zeros(N, 1);
    chunk = 80;
    for b = 1:chunk:N
        fin = min(b + chunk - 1, N); loc_str = strjoin(arrayfun(@(idx) sprintf('%f,%f', lat(idx), lon(idx)), b:fin, 'UniformOutput', false), '|'); 
        try res = webread(sprintf('https://api.opentopodata.org/v1/srtm30m?locations=%s', loc_str), weboptions('Timeout', 10)); if isfield(res, 'results'), Alt_m(b:fin) = arrayfun(@(x) x.elevation, res.results); end; catch, Alt_m(b:fin) = City.Alt; end
    end
    R_earth = 6371000; X_m = deg2rad(lon - City.Lon) * R_earth * cosd(City.Lat); Y_m = deg2rad(lat - City.Lat) * R_earth;
    Clutter = strings(N, 1);
    for i = 1:N, neighbors = sum(sqrt((X_m - X_m(i)).^2 + (Y_m - Y_m(i)).^2) < 300);
        if neighbors > 40, Clutter(i) = "Dense Urban"; elseif neighbors > 15, Clutter(i) = "Residential"; else, Clutter(i) = "Park/Periphery"; end
    end
    T = unique(table(Alt_m, lat, lon, X_m, Y_m, Clutter, 'VariableNames', {'Altitude_m', 'Lat', 'Lon', 'X_m', 'Y_m', 'Clutter'}), 'rows');
end

function RSSI_Matrix = Precompute_RSSI(T, HW)
    N = height(T); RSSI_Matrix = zeros(N,N) - 150; L0 = 20*log10(HW.Freq) - 27.55; 
    for gw = 1:N
        d_m = sqrt((T.X_m - T.X_m(gw)).^2 + (T.Y_m - T.Y_m(gw)).^2 + (T.Altitude_m - T.Altitude_m(gw)).^2); d_m(d_m < 1) = 1; 
        n_path = 3.2 * ones(N, 1); L_clut = zeros(N,1);
        idx_den = T.Clutter == "Dense Urban"; L_clut(idx_den) = 18; n_path(idx_den) = 3.8; 
        idx_prk = T.Clutter == "Park/Periphery"; L_clut(idx_prk) = 12; n_path(idx_prk) = 3.5; 
        Alt_GW_Real = T.Altitude_m(gw) + HW.Height; diff_alt = Alt_GW_Real - T.Altitude_m; 
        L_topo = zeros(N, 1); v_idx = diff_alt > 15; L_topo(v_idx) = min(15 * log10(diff_alt(v_idx)), 35); 
        c_idx = diff_alt < -15; L_topo(c_idx) = min(10 * log10(abs(diff_alt(c_idx))), 35); 
        RSSI_Matrix(:, gw) = (HW.Ptx + HW.Gain) - L0 - 10 .* n_path .* log10(d_m) - L_clut - HW.PenetrationLoss - L_topo;
    end
end

function sol = Run_Greedy(BinM_3D, target)
    [N, C] = size(BinM_3D); sol = false(1, C); cov = false(N, 1);
    while (sum(cov)/N) < target
        b_g = 0; b_n = -1; cands = find(~sol);
        for i = cands, g = sum(BinM_3D(:, i) & ~cov); if g>b_g, b_g=g; b_n=i; end, end
        if b_n > 0, sol(b_n) = true; cov = cov | BinM_3D(:, b_n); else, break; end
    end
end

function sol = Run_KMedoids(Pts_3D, BinM_3D, target)
    N_c = size(BinM_3D, 2); sol = false(1, N_c); max_k = 30; 
    for k = 1:2:max_k 
        [~, C] = kmedoids(Pts_3D, k, 'Distance', 'euclidean', 'Replicates', 1, 'Options', statset('UseParallel', false));
        idx_m = zeros(1, k); for i=1:k, [~, real_i] = min(sum((Pts_3D - C(i,:)).^2, 2)); idx_m(i) = real_i; end
        if (sum(sum(BinM_3D(:, idx_m), 2) > 0) / size(BinM_3D, 1)) >= target, sol(idx_m) = true; break; end
    end
end

function sol = Run_GA(BinM_3D, target, use_memetic)
    [N, C] = size(BinM_3D); p_size = 24; gen = 30; Pop = rand(p_size, C) < (10/C); Pop(1,:) = Run_Greedy(BinM_3D, target); 
    for g = 1:gen
        Fit = zeros(p_size, 1);
        for i = 1:p_size
            mapa = sum(BinM_3D(:, find(Pop(i,:))), 2); c_tot = sum(mapa >= 1)/N; c_k2 = sum(mapa >= 2)/N; gw_cnt = sum(Pop(i,:));
            Fit(i) = (c_tot * 10000) + (c_k2 * 1000) - (gw_cnt * 50);
            if c_tot < target, Fit(i) = Fit(i) - 50000; end
        end
        [~, b] = max(Fit); Best = Pop(b,:); NewPop = Pop;
        for i = 2:2:p_size
            p1 = Pop(randi(p_size),:); p2 = Pop(randi(p_size),:); mask = rand(1, C) > 0.5; c1 = p1; c1(mask) = p2(mask);
            if rand < 0.8, if use_memetic, c1 = Smart_Repair(c1, BinM_3D, target); else, mut = randi(C); c1(mut) = ~c1(mut); end; end
            NewPop(i,:) = c1;
        end
        Pop = NewPop; Pop(1,:) = Best;
    end
    sol = Pop(1,:);
end

function ind = Smart_Repair(ind, BinM_3D, target)
    N = size(BinM_3D, 1); max_iter = 10; iter = 0;
    while iter < max_iter
        iter = iter + 1; idx = find(ind); mapa = sum(BinM_3D(:, idx), 2);
        if (sum(mapa >= 1) / N) >= target
            if length(idx) > 1, unq = (mapa == 1); sc = sum(BinM_3D(:, idx) & unq, 1); [val, w] = min(sc); if val < 5, ind(idx(w)) = false; continue; end; end; break;
        else
            unc = (mapa == 0); cands = find(~ind); if isempty(cands), break; end
            gains = sum(BinM_3D(unc, cands), 1); [bg, best_idx] = max(gains);
            if bg > 0, bn = cands(best_idx); ind(bn) = true; else, break; end
        end
    end
end