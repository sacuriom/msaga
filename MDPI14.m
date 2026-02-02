function MDPI14
    % =====================================================================
    % TESIS DOCTORAL: LORAWAN BENCHMARK DASHBOARD - V14 (ASCII CLEAN)
    % Autor: Tesis MDPI
    % Fix: Eliminados todos los caracteres no-ASCII (tildes, emojis).
    %      Garantiza compatibilidad total con el editor.
    % =====================================================================
    clc; close all; warning('off','all');

    %% 1. CONFIGURACION DEL ESTUDIO
    cfg.N_runs = 50; 
    
    % Base de Datos de Ciudades
    Cities(1) = struct('Name','Ambato, ECU',    'Lat',-1.2417, 'Lon',-78.6197, 'Zoom',15, 'T_GW',12);
    Cities(2) = struct('Name','Quito, ECU',     'Lat',-0.1807, 'Lon',-78.4678, 'Zoom',14, 'T_GW',18);
    Cities(3) = struct('Name','Guayaquil, ECU', 'Lat',-2.1894, 'Lon',-79.8891, 'Zoom',14, 'T_GW',20);
    Cities(4) = struct('Name','Cuenca, ECU',    'Lat',-2.9001, 'Lon',-79.0059, 'Zoom',15, 'T_GW',14);
    Cities(5) = struct('Name','New York, USA',  'Lat',40.7128, 'Lon',-74.0060, 'Zoom',14, 'T_GW',30);
    Cities(6) = struct('Name','Madrid, ESP',    'Lat',40.4168, 'Lon',-3.7038,  'Zoom',14, 'T_GW',25);

    % Parametros
    cfg.apiKey = 'AIzaSyAnWExFoChjUJSrynNmAqKgNutZ9cL4Eq8'; 
    cfg.pop_size = 40; 
    cfg.max_iter = 50; 
    cfg.w_cov = 2.5; cfg.w_ovr = 0.1; cfg.w_cost = 0.5;

    %% 2. INTERFAZ GRAFICA
    fig = uifigure('Name', 'LoRaWAN Monte Carlo Benchmark', 'Position', [50 50 1100 750]);
    g = uigridlayout(fig, [2, 2]);
    g.RowHeight = {'1x', 250}; g.ColumnWidth = {300, '1x'};

    % --- EJE DEL MAPA ---
    ax_map = uiaxes(g);
    ax_map.Layout.Row = 1; ax_map.Layout.Column = 2;
    title(ax_map, 'Visualizacion Mejor Solucion (M-SAGA)');
    ax_map.XLabel.String = 'Longitud'; ax_map.YLabel.String = 'Latitud';
    grid(ax_map, 'on');

    % --- TABLA DE RESULTADOS ---
    tbl_kpi = uitable(g);
    tbl_kpi.Layout.Row = 2; tbl_kpi.Layout.Column = [1 2];
    tbl_kpi.ColumnName = {'CIUDAD', 'ALGORITMO', 'COBERTURA (%)', 'EFICIENCIA (Ptos/GW)', 'ROBUSTEZ', 'CONVERGENCIA (Gen)', 'TIEMPO (s)'};
    tbl_kpi.RowName = {};

    % --- PANEL DE CONTROL ---
    pnl = uipanel(g, 'Title', 'Configuracion Monte Carlo');
    pnl.Layout.Row = 1; pnl.Layout.Column = 1;
    
    uilabel(pnl, 'Text', 'Ciudad Objetivo:', 'Position', [20 350 200 20], 'FontWeight', 'bold');
    
    ef_gw = uieditfield(pnl, 'numeric', 'Position', [20 250 100 30], 'Value', Cities(1).T_GW);
    
    dd_city = uidropdown(pnl, 'Items', {Cities.Name}, 'Position', [20 320 240 30], ...
        'ValueChangedFcn', @(src,event) update_gw_field(src, Cities, ef_gw));
    
    uilabel(pnl, 'Text', 'Gateways Estimados:', 'Position', [20 280 150 20]);
    
    uilabel(pnl, 'Text', ['Simulaciones por Algo: N=' num2str(cfg.N_runs)], ...
        'Position', [20 200 250 20], 'FontColor', 'blue');
    
    uilabel(pnl, 'Text', 'Estado:', 'Position', [20 160 100 20]);
    lbl_status = uilabel(pnl, 'Text', 'Listo', 'Position', [20 140 240 20], 'FontColor', [0.4 0.4 0.4]);
    
    % Boton de Ejecucion
    uibutton(pnl, 'Text', 'INICIAR BENCHMARK', ...
        'Position', [20 80 240 50], 'BackgroundColor', [0.6 0.2 0.2], 'FontColor', 'w', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(btn,event) run_montecarlo(dd_city, ef_gw, Cities, cfg, ax_map, tbl_kpi, fig, lbl_status));

    setappdata(fig, 'History', {});
end

function update_gw_field(dd, StructC, ef)
    idx = strcmp({StructC.Name}, dd.Value); ef.Value = StructC(idx).T_GW;
end

%% ========================================================================
%% LOGICA MONTE CARLO
%% ========================================================================
function run_montecarlo(dd, ef_gw, StructC, cfg, ax, tbl, fig, lbl)
    idx = strcmp({StructC.Name}, dd.Value);
    City = StructC(idx);
    
    cfg.ciudad_nombre = City.Name;
    cfg.lat_center = City.Lat; cfg.lon_center = City.Lon;
    cfg.zoom = City.Zoom; cfg.target_gateways = ef_gw.Value;
    cfg.max_gateways = round(ef_gw.Value * 1.3);
    
    % 1. Adquisicion de Datos
    lbl.Text = 'Descargando Topologia...'; drawnow;
    d = uiprogressdlg(fig, 'Title', 'Fase 1: Adquisicion', 'Indeterminate', 'on');
    
    try
        [raw, img, R, cfg] = Fase1_Acquisition(cfg);
        if isempty(raw), error('Fallo descarga OSM.'); end
        
        cand = Fase2_Filter(raw, 0.025);
        coords = [cand.X_km, cand.Y_km];
        DistM = pdist2(coords, coords);
        BinM = DistM <= cfg.radio_km;
        
        % FIX: Nombres validos para struct
        AlgosKeys = {'Greedy', 'B_PSO', 'M_SAGA'}; 
        AlgosDisplay = {'Greedy', 'B-PSO', 'M-SAGA'};
        Stats = struct();
        
        % 2. Bucle Monte Carlo
        d.Indeterminate = 'off';
        total_steps = cfg.N_runs * 3;
        curr_step = 0;
        
        Median_Cov = -1;
        Best_Sol_Visual = [];
        
        for a = 1:3
            AlgoKey = AlgosKeys{a};     
            AlgoName = AlgosDisplay{a}; 
            
            res_cov = []; res_eff = []; res_rob = []; res_time = []; res_gen = [];
            
            for i = 1:cfg.N_runs
                curr_step = curr_step + 1;
                d.Value = curr_step / total_steps;
                d.Message = sprintf('Simulando %s: Iteracion %d/%d', AlgoName, i, cfg.N_runs);
                
                tic;
                if strcmp(AlgoKey, 'Greedy')
                    [sol, gen] = Solver_Greedy(BinM, cfg);
                elseif strcmp(AlgoKey, 'B_PSO')
                    [sol, gen] = Solver_BPSO(BinM, cfg);
                else
                    [sol, gen] = Solver_MSAGA(BinM, cfg);
                end
                t = toc;
                
                K = Calc_KPIs(sol, cand, t, BinM);
                
                res_cov(end+1) = K.Cov;
                res_eff(end+1) = K.Eff;
                res_rob(end+1) = K.Rob;
                res_time(end+1) = t;
                res_gen(end+1) = gen;
                
                if strcmp(AlgoKey, 'M_SAGA')
                    if K.Cov > Median_Cov 
                        Median_Cov = K.Cov;
                        Best_Sol_Visual = K;
                    end
                end
            end
            
            % Guardar
            Stats.(AlgoKey).Cov = mean(res_cov);
            Stats.(AlgoKey).Eff = mean(res_eff);
            Stats.(AlgoKey).Rob = mean(res_rob);
            Stats.(AlgoKey).Time = mean(res_time);
            Stats.(AlgoKey).Gen = mean(res_gen);
        end
        
        % 3. Actualizar Tabla
        hist = getappdata(fig, 'History');
        for a = 1:3
            Key = AlgosKeys{a};
            Name = AlgosDisplay{a};
            S = Stats.(Key);
            % Fila: {Ciudad, Algoritmo, Cov, Eff, Rob, Conv_Gen, Time}
            row = {City.Name, Name, S.Cov, S.Eff, S.Rob, S.Gen, S.Time};
            hist = [hist; row];
        end
        setappdata(fig, 'History', hist);
        tbl.Data = hist;
        
        % 4. Plotear Mejor Caso
        if ~isempty(Best_Sol_Visual)
            Plot_Map(ax, img, R, cand, Best_Sol_Visual, cfg);
        end
        
        lbl.Text = 'Benchmark Completado';
        close(d);
        
    catch ME
        close(d);
        uialert(fig, ME.message, 'Error Fatal');
    end
end

%% ========================================================================
%% ALGORITMOS
%% ========================================================================
function [sol, conv_gen] = Solver_Greedy(BinM, cfg)
    [~, N] = size(BinM); sol = false(1,N); cov = false(size(BinM,1),1);
    step = 0;
    for k=1:cfg.target_gateways
        best_g=-1; best_n=-1; cands=find(~sol);
        for i=cands
            g=sum(BinM(:,i)&~cov); 
            if g>best_g, best_g=g; best_n=i; end
        end
        if best_n>0
            sol(best_n)=true; cov=cov|BinM(:,best_n); step=step+1;
        else, break; end
    end
    conv_gen = step; 
end

function [sol, conv_gen] = Solver_BPSO(BinM, cfg)
    [~, N] = size(BinM);
    X = rand(cfg.pop_size, N) < (cfg.target_gateways/N);
    V = zeros(cfg.pop_size, N);
    Pbest_X=X; Pbest_F=-inf(cfg.pop_size,1);
    Gbest_F=-inf; Gbest_X=X(1,:);
    conv_gen = 1;
    
    for i=1:cfg.pop_size, Pbest_F(i)=Fitness(X(i,:),BinM,cfg); end
    [Gbest_F,idx]=max(Pbest_F); Gbest_X=Pbest_X(idx,:);
    
    for t=1:cfg.max_iter
        improved = false;
        w=0.9-0.5*(t/cfg.max_iter);
        for i=1:cfg.pop_size
            f = Fitness(X(i,:),BinM,cfg);
            if f>Pbest_F(i), Pbest_F(i)=f; Pbest_X(i,:)=X(i,:);
                if f>Gbest_F, Gbest_F=f; Gbest_X=X(i,:); improved=true; end
            end
            r1=rand(1,N); r2=rand(1,N);
            V(i,:) = w*V(i,:) + 2*r1.*(Pbest_X(i,:)-X(i,:)) + 2*r2.*(Gbest_X-X(i,:));
            S = 1./(1+exp(-V(i,:))); X(i,:) = rand(1,N) < S;
        end
        if improved, conv_gen = t; end 
    end
    sol = Gbest_X;
end

function [sol, conv_gen] = Solver_MSAGA(BinM, cfg)
    [~, N] = size(BinM);
    Pop = rand(cfg.pop_size, N) < (cfg.target_gateways/N);
    Pop(1,:) = Solver_Greedy(BinM, cfg); 
    
    Fit = zeros(cfg.pop_size,1);
    conv_gen = 1;
    BestFit_Global = -inf;
    
    for g=1:cfg.max_iter
        for i=1:cfg.pop_size, Fit(i)=Fitness(Pop(i,:),BinM,cfg); end
        [curr_best_val, b] = max(Fit);
        Best = Pop(b,:);
        
        if curr_best_val > BestFit_Global
            BestFit_Global = curr_best_val;
            conv_gen = g; 
        end
        
        NewPop = Pop;
        for i=2:2:cfg.pop_size
            r=randi(cfg.pop_size,1,2); [~,w1]=max(Fit(r)); p1=Pop(r(w1),:);
            r=randi(cfg.pop_size,1,2); [~,w2]=max(Fit(r)); p2=Pop(r(w2),:);
            mask=rand(1,N)>0.5; c1=p1; c1(mask)=p2(mask); c2=p2; c2(mask)=p1(mask);
            if rand<0.6, c1=Mutate(c1,BinM,cfg); end
            if rand<0.6, c2=Mutate(c2,BinM,cfg); end
            NewPop(i,:)=c1; if i+1<=cfg.pop_size, NewPop(i+1,:)=c2; end
        end
        Pop=NewPop; Pop(1,:)=Best;
    end
    sol = Pop(1,:);
end

function f = Fitness(ind, BinM, cfg)
    idx=ind; n=sum(idx); if n==0, f=-inf; return; end
    cnt=sum(BinM(:,idx),2); cov=sum(cnt>=1); ovr=sum(cnt>1);
    pen=0; if n>cfg.target_gateways, pen=(n-cfg.target_gateways)*50; end
    f = (cfg.w_cov*cov) - (cfg.w_ovr*ovr) - (cfg.w_cost*pen);
end

function ind = Mutate(ind, BinM, cfg)
    idx=find(ind); n=length(idx);
    if n >= cfg.max_gateways || (n > cfg.target_gateways && rand<0.5)
        sc=sum(sum(BinM(:,idx),2)==1,1); [~,w]=min(sc); ind(idx(w))=false;
    else
        unc=find(sum(BinM(:,idx),2)==0);
        if ~isempty(unc)
            cands=find(~ind);
            if length(cands)>30, cands=cands(randperm(length(cands),30)); end
            bg=-1; bn=-1;
            for c=cands, g=sum(BinM(unc,c)); if g>bg, bg=g; bn=c; end, end
            if bn>0, ind(bn)=true; end
        end
    end
end

%% ========================================================================
%% HELPERS Y FASE 1
%% ========================================================================
function K = Calc_KPIs(sol, T, t, BinM)
    idx=find(sol); cnt=sum(BinM(:,idx),2);
    K.NumGW=length(idx); K.Cov=(sum(cnt>=1)/height(T))*100;
    if K.NumGW>0, K.Eff=sum(cnt>=1)/K.NumGW; else, K.Eff=0; end
    if sum(cnt>=1)>0, K.Rob=sum(cnt(cnt>=1))/sum(cnt>=1); else, K.Rob=0; end
    K.Huecos=(cnt==0); K.Sol=sol;
end

function Plot_Map(ax, img, R, T, K, cfg)
    cla(ax); imshow(img, R, 'Parent', ax); set(ax, 'YDir', 'normal'); hold(ax, 'on');
    scatter(ax, T.Lon(K.Huecos), T.Lat(K.Huecos), 15, 'r', 'filled');
    scatter(ax, T.Lon(~K.Huecos), T.Lat(~K.Huecos), 5, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha',0.5);
    idx = find(K.Sol); theta = linspace(0, 2*pi, 25);
    dlat = cfg.radio_km/111.32; dlon = cfg.radio_km/(111.32*cosd(cfg.lat_center));
    for i=1:length(idx)
        plot(ax, T.Lon(idx(i))+dlon*cos(theta), T.Lat(idx(i))+dlat*sin(theta), 'b-', 'LineWidth', 1.5);
    end
    scatter(ax, T.Lon(idx), T.Lat(idx), 60, 'k', 'filled', 'MarkerEdgeColor', 'w');
end

function [T_out, img, R_ref, cfg_out] = Fase1_Acquisition(cfg)
    cfg_out = cfg; sz = 640;
    style = '&style=feature:all|element:labels|visibility:off';
    url = sprintf('https://maps.googleapis.com/maps/api/staticmap?center=%f,%f&zoom=%d&size=%dx%d&maptype=roadmap&key=%s%s', ...
        cfg.lat_center, cfg.lon_center, cfg.zoom, sz, sz, cfg.apiKey, style);
    try img = flipud(webread(url)); catch, img = ones(sz,sz)*240; end
    nTiles = 2^cfg.zoom;
    m_px = 156543.03392 * cosd(cfg.lat_center) / nTiles;
    dlat = (m_px/1000)/111.32; dlon = (m_px/1000)/(111.32*cosd(cfg.lat_center));
    LatLim = [cfg.lat_center - (sz/2)*dlat, cfg.lat_center + (sz/2)*dlat];
    LonLim = [cfg.lon_center - (sz/2)*dlon, cfg.lon_center + (sz/2)*dlon];
    R_ref = imref2d(size(img), LonLim, LatLim);
    roi = polyshape([LonLim(1), LonLim(2), LonLim(2), LonLim(1)], [LatLim(1), LatLim(1), LatLim(2), LatLim(2)]);
    T_OSM = Fetch_OSM(LatLim(1)-0.003, LonLim(1)-0.003, LatLim(2)+0.003, LonLim(2)+0.003);
    if isempty(T_OSM), T_out=[]; return; end
    in_roi = isinterior(roi, T_OSM.Lon, T_OSM.Lat); T_OSM = T_OSM(in_roi, :);
    R_earth = 6371;
    x_km = deg2rad(T_OSM.Lon)*R_earth.*cosd(mean(T_OSM.Lat));
    y_km = deg2rad(T_OSM.Lat)*R_earth;
    if contains(cfg.ciudad_nombre, {'New York','Madrid','Berlin'}), r=0.4;
    elseif contains(cfg.ciudad_nombre, {'Quito','Guayaquil'}), r=0.6; else, r=0.8; end
    cfg_out.radio_km = r;
    T_out = table(T_OSM.Lat, T_OSM.Lon, x_km, y_km, T_OSM.Tipo, 'VariableNames', {'Lat','Lon','X_km','Y_km','Tipo'});
end

function T = Fetch_OSM(min_lat, min_lon, max_lat, max_lon)
    srv = {'https://overpass-api.de/api/interpreter', 'https://lz4.overpass-api.de/api/interpreter'};
    q = sprintf('[out:json][timeout:90];(way["highway"~"^(primary|secondary|tertiary|residential)$"](%f,%f,%f,%f););out body;>;out skel qt;', ...
        min_lat, min_lon, max_lat, max_lon);
    data=[]; 
    for s=1:2, try data=webwrite(srv{s},'data',q,weboptions('Timeout',90)); break; catch, continue; end, end
    if isempty(data)||~isfield(data,'elements'), T=[]; return; end
    raw = data.elements; c = containers.Map('KeyType','double','ValueType','any');
    cnt = containers.Map('KeyType','double','ValueType','double');
    for i=1:length(raw), if iscell(raw), e=raw{i}; else, e=raw(i); end, if strcmp(e.type,'node'), c(e.id)=[e.lat, e.lon]; end, end
    lat=[]; lon=[]; tip=[];
    for i=1:length(raw)
        if iscell(raw), e=raw{i}; else, e=raw(i); end
        if strcmp(e.type,'way') && ~isempty(e.nodes)
            nds=e.nodes;
            for k=1:length(nds), if isKey(cnt,nds(k)), cnt(nds(k))=cnt(nds(k))+1; else, cnt(nds(k))=1; end, end
            for k=2:length(nds)
                if isKey(c,nds(k)) && isKey(c,nds(k-1))
                   p1=c(nds(k)); p2=c(nds(k-1)); d=sqrt(sum((p1-p2).^2))*111320;
                   if d>40
                       np=floor(d/30);
                       for p=1:np, fr=p/(np+1); lat=[lat;p2(1)+(p1(1)-p2(1))*fr]; lon=[lon;p2(2)+(p1(2)-p2(2))*fr]; tip=[tip;0]; end
                   end
                end
            end
        end
    end
    keys_c=cell2mat(cnt.keys); vals_c=cell2mat(cnt.values); int_ids=keys_c(vals_c>=2);
    lat_int=zeros(length(int_ids),1); lon_int=zeros(length(int_ids),1); val=false(length(int_ids),1);
    for k=1:length(int_ids), if isKey(c,int_ids(k)), cv=c(int_ids(k)); lat_int(k)=cv(1); lon_int(k)=cv(2); val(k)=true; end, end
    lat=[lat; lat_int(val)]; lon=[lon; lon_int(val)]; tip=[tip; ones(sum(val),1)];
    T = unique(table(lat, lon, tip, 'VariableNames', {'Lat','Lon','Tipo'}));
end

function T_out = Fase2_Filter(T_raw, rad)
    data=[T_raw.X_km, T_raw.Y_km]; idx=rangesearch(createns(data),data,rad); kp=true(height(T_raw),1);
    tip=T_raw.Tipo;
    for i=1:height(T_raw)
        if kp(i), n=idx{i}; n(n==i)=[]; 
            if tip(i)==0 && any(tip(n)==1), kp(i)=false; continue; end
            kp(n)=false; 
        end
    end
    T_out=T_raw(kp,:); 
end