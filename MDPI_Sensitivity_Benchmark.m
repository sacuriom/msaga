function MDPI_Sensitivity_Benchmark
    % =====================================================================
    % DOCTORAL THESIS: 3D PARAMETRIC SENSITIVITY BENCHMARK
    % Sweep: Heights [5, 10, 15] meters x SF [10, 11, 12]
    % Generates Master CSV and Global HTML Report in a specific folder
    % =====================================================================
    clc; close all; warning('off', 'all');
    
    disp('======================================================');
    disp(' INITIALIZING MASSIVE BENCHMARK (PARAMETRIC GRID SEARCH) ');
    disp('======================================================');
    
    %% 1. CITY SELECTION & BASE PARAMETERS
    Cities(1) = struct('Name','Ambato, ECU', 'Lat',-1.2417, 'Lon',-78.6197, 'Alt',2580);
    Cities(2) = struct('Name','Cuenca, ECU', 'Lat',-2.9001, 'Lon',-79.0059, 'Alt',2560);
    Cities(3) = struct('Name','Quito, ECU',  'Lat',-0.1807, 'Lon',-78.4678, 'Alt',2850);
    
    [indx, tf] = listdlg('PromptString', 'Select the Target City for Benchmark:', ...
                         'SelectionMode', 'single', 'ListString', {Cities.Name}, 'ListSize', [250 120]);
    if ~tf, disp('Operation cancelled by user.'); return; end
    City = Cities(indx);
    
    Target_Cov = 0.98; % Strict 98% coverage target
    
    HW.Name = 'Milesight UG67 Outdoor';
    HW.Freq = 915;          
    HW.Ptx = 27;            
    HW.Gain = 5;            
    HW.PenetrationLoss = 0; 
    
    H_list = [5, 10, 15];      % Antenna heights to evaluate (meters)
    SF_list = [10, 11, 12];    % Spreading Factors to evaluate
    Sens_Map = containers.Map({7,8,9,10,11,12}, {-125, -128, -131, -134, -136, -137});
    
    Algo_Names = ["Greedy SCP", "K-Medoids 3D", "Standard SAGA", "Proposed 3D-M-SAGA"];
    
    % Prepare Master Results Table with English variables
    VarNames = {'Height_m', 'SF', 'Sensitivity_dBm', 'Algorithm', 'Gateways', 'Coverage_Pct', 'K1_Risk', 'K2_Ideal', 'K3_Excess', 'Time_s'};
    VarTypes = {'double', 'double', 'double', 'string', 'double', 'double', 'double', 'double', 'double', 'double'};
    Master_Results = table('Size', [0, length(VarNames)], 'VariableTypes', VarTypes, 'VariableNames', VarNames);
    
    % Dynamic Folder Creation
    city_clean = strrep(strrep(City.Name, ', ', '_'), ' ', ''); 
    fecha_hora = datestr(now, 'yyyymmdd_HHMMSS');
    folder_name = sprintf('Benchmark_%s_%s', city_clean, fecha_hora);
    mkdir(folder_name);
    
    %% 2. BUILD DIGITAL TWIN 
    disp(' -> [PHASE 1] Building Base 3D Digital Twin...');
    T_Twin = Build_DigitalTwin(City, 0.020); 
    if isempty(T_Twin), return; end
    Pts_3D = [T_Twin.X_m, T_Twin.Y_m, T_Twin.Altitude_m];
    
    %% 3. MAIN LOOP (GRID SEARCH)
    total_runs = length(H_list) * length(SF_list);
    current_run = 0;
    
    for h = H_list
        disp(' ');
        fprintf('======================================================\n');
        fprintf(' EVALUATING ANTENNA HEIGHT: %d METERS \n', h);
        fprintf('======================================================\n');
        HW.Height = h;
        
        % Precompute physical matrix only once per height to save time
        disp(' -> Computing Topographic Diffraction and 3D Clutter...');
        RSSI_Matrix = Precompute_RSSI(T_Twin, HW);
        
        for sf = SF_list
            current_run = current_run + 1;
            HW.SF = sf; HW.Sens = Sens_Map(sf);
            fprintf('\n   [Scenario %d/%d] Height: %dm | SF: %d (Sens: %d dBm)\n', current_run, total_runs, h, sf, HW.Sens);
            
            % Immediate boolean filter based on SF sensitivity
            BinM_3D = RSSI_Matrix >= HW.Sens;
            
            % Target adjustment if physical limits are too harsh
            max_cov = sum(sum(BinM_3D, 2) > 0) / height(T_Twin);
            if max_cov < Target_Cov
                fprintf('      [!] Warning: Physical limit at %.2f%%. Adjusting target.\n', max_cov*100);
                Actual_Target = max_cov * 0.99;
            else
                Actual_Target = Target_Cov;
            end
            
            % Execute the 4 algorithms
            for a = 1:4
                tic;
                if a == 1
                    sol = Run_Greedy(BinM_3D, Actual_Target);
                elseif a == 2
                    sol = Run_KMedoids(Pts_3D, BinM_3D, Actual_Target);
                elseif a == 3
                    sol = Run_GA(BinM_3D, Actual_Target, false);
                elseif a == 4
                    sol = Run_GA(BinM_3D, Actual_Target, true);
                end
                t_exec = toc;
                
                % Extract KPIs and append row
                row_data = Extract_KPIs(h, sf, HW.Sens, Algo_Names(a), sol, BinM_3D, t_exec);
                Master_Results = [Master_Results; row_data];
            end
        end
    end
    
    %% 4. EXPORT RESULTS (CSV & HTML)
    disp(' ');
    disp('======================================================');
    disp(' -> PARAMETRIC SWEEP SUCCESSFULLY COMPLETED! ');
    disp(' -> Generating Global Reports...');
    disp('======================================================');
    
    % Export Master CSV
    file_csv = fullfile(folder_name, 'Master_Sensitivity_Results.csv');
    writetable(Master_Results, file_csv);
    
    % Generate Dynamic HTML Report
    Generate_Benchmark_HTML_Report(City, HW, Target_Cov, Master_Results, folder_name, H_list, SF_list, Algo_Names);
    
    disp([' -> Check results in folder: ', folder_name]);
end

%% ========================================================================
%% INTERNAL FUNCTIONS (TWIN, RF, ALGORITHMS)
%% ========================================================================
function T = Build_DigitalTwin(City, dist)
    min_lat = City.Lat - dist; min_lon = City.Lon - dist; max_lat = City.Lat + dist; max_lon = City.Lon + dist;
    options = weboptions('Timeout', 120); 
    q_osm = sprintf('[out:json][timeout:120];(way["highway"~"^(primary|secondary|tertiary|residential)$"](%f,%f,%f,%f););out body;>;out skel qt;', min_lat, min_lon, max_lat, max_lon);
    try data = webread('https://lz4.overpass-api.de/api/interpreter', 'data', q_osm, options); catch, T=[]; disp('OSM Error'); return; end
    
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

function row = Extract_KPIs(h, sf, sens, name, sol, BinM_3D, time_s)
    idx = find(sol); n_gws = length(idx); N = size(BinM_3D, 1);
    if n_gws == 0, row = {h, sf, sens, name, 0, 0, 0, 0, 0, round(time_s,2)}; return; end
    mapa = sum(BinM_3D(:, idx), 2); c_tot = round((sum(mapa >= 1)/N)*100, 2);
    k1 = round((sum(mapa == 1)/N)*100, 2); k2 = round((sum(mapa == 2)/N)*100, 2); k3 = round((sum(mapa >= 3)/N)*100, 2);
    row = {h, sf, sens, name, n_gws, c_tot, k1, k2, k3, round(time_s, 2)};
end

%% ========================================================================
%% GLOBAL HTML REPORT GENERATOR
%% ========================================================================
function Generate_Benchmark_HTML_Report(City, HW, Target_Cov, Results, folder, H_list, SF_list, Algo_Names)
    file = fullfile(folder, 'Global_Sensitivity_Benchmark_Report.html'); fid = fopen(file, 'w', 'n', 'UTF-8');
    
    fprintf(fid, '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{font-family:Arial, sans-serif;margin:40px;color:#333;line-height:1.6;} table{border-collapse:collapse;width:100%%;margin-bottom:40px;} th,td{border:1px solid #ccc;padding:8px;text-align:center;font-size:13px;} th{background-color:#003366;color:white;} .box{background-color:#f0f4f8;padding:20px;border-left:5px solid #003366;margin-bottom:20px;} h2{color:#003366;border-bottom:2px solid #ccc;padding-bottom:5px;} .h-title{background-color:#e0e0e0; font-weight:bold;} .algo-row{font-weight:bold;}</style></head><body>\n');
    fprintf(fid, '<h1>LoRaWAN 3D Sensitivity Analysis & Benchmark Report</h1>\n');
    fprintf(fid, '<div class="box"><p><strong>City:</strong> %s</p><p><strong>Hardware Base:</strong> %s (Tx: %.1f dBm, Gain: %.1f dBi)</p><p><strong>Target Coverage:</strong> %.1f%%</p></div>\n', City.Name, HW.Name, HW.Ptx, HW.Gain, Target_Cov*100);
    
    fprintf(fid, '<h2>Comprehensive Parametric Sweep Results</h2>\n');
    fprintf(fid, '<p>This table displays the algorithm performance across different physical constraints (Antenna Heights) and Spreading Factors (SF).</p>\n');
    
    for h = H_list
        fprintf(fid, '<h3>Analysis for Antenna Height: %d meters</h3>\n', h);
        fprintf(fid, '<table><tr><th>Algorithm</th><th>Spreading Factor</th><th>Sensitivity</th><th>Gateways</th><th>Coverage</th><th>K=1 (Risk)</th><th>K=2 (Ideal)</th><th>K&ge;3 (Excess)</th><th>Time (s)</th></tr>\n');
        
        T_H = Results(Results.Height_m == h, :);
        
        for sf = SF_list
            T_SF = T_H(T_H.SF == sf, :);
            
            for a = 1:length(Algo_Names)
                row = T_SF(T_SF.Algorithm == Algo_Names(a), :);
                if isempty(row), continue; end
                
                % Highlight high risk (>25%) in Red and M-SAGA in Blue
                color_k1 = 'color:#333;'; if row.K1_Risk > 25, color_k1 = 'color:#d32f2f; font-weight:bold;'; end
                color_gw = 'color:#333;'; if a == 4, color_gw = 'color:#1565c0; font-weight:bold;'; end 
                
                fprintf(fid, '<tr><td class="algo-row">%s</td><td>SF%d</td><td>%.1f dBm</td><td style="%s">%d</td><td>%.2f%%</td><td style="%s">%.2f%%</td><td>%.2f%%</td><td>%.2f%%</td><td>%.2f</td></tr>\n', ...
                    row.Algorithm(1), row.SF(1), row.Sensitivity_dBm(1), color_gw, row.Gateways(1), row.Coverage_Pct(1), color_k1, row.K1_Risk(1), row.K2_Ideal(1), row.K3_Excess(1), row.Time_s(1));
            end
        end
        fprintf(fid, '</table>\n');
    end
    
    fprintf(fid, '</body></html>\n'); fclose(fid);
end