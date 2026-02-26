function MDPI_Morfologico_Milesight_V33_Consistente
    % =====================================================================
    % TESIS DOCTORAL: GEMELO MORFOLÓGICO & SIMULACIÓN MILESIGHT (V33.2)
    % Novedad: Consistencia absoluta 2D/3D (Tooltips Universales, DMS, ID Gateways)
    % =====================================================================
    clc; close all; warning('off','all');
    
    %% 1. BASE DE DATOS Y HARDWARE
    cfg.N_runs = 1; 
    cfg.pop_size = 40; cfg.max_iter = 50; 
    cfg.w_cov = 2.5; cfg.w_ovr = 0.1; cfg.w_cost = 0.5;
    cfg.target_gateways = 8;
    cfg.max_gateways = round(cfg.target_gateways * 1.3);
    
    Cities(1) = struct('Name','Ambato, ECU', 'Lat',-1.2417, 'Lon',-78.6197, 'Alt',2580);
    Cities(2) = struct('Name','Quito, ECU',  'Lat',-0.1807, 'Lon',-78.4678, 'Alt',2850);
    Cities(3) = struct('Name','Cuenca, ECU', 'Lat',-2.9001, 'Lon',-79.0059, 'Alt',2560);
    
    % Hardware Real
    cfg.Models = struct('Name', {'Milesight UG67 (Outdoor)', 'Milesight UG65 (Indoor)'}, ...
                        'Ptx', {27, 24}, 'Gain', {5, 2}, 'Sens', {-147, -140}, ...
                        'PenetrationLoss', {0, 15}); 
                        
    %% 2. INTERFAZ GRÁFICA 
    fig = uifigure('Name', 'Validación Morfológica Milesight vs M-SAGA', 'Position', [50 50 1200 800]);
    g = uigridlayout(fig, [2, 2]); g.RowHeight = {'1x', 200}; g.ColumnWidth = {300, '1x'};
    
    ax_map = uiaxes(g); ax_map.Layout.Row = 1; ax_map.Layout.Column = 2;
    title(ax_map, 'Mapa de Calor RSSI: Validación Física Milesight');
    
    pnl = uipanel(g, 'Title', 'Control del Gemelo Morfológico');
    pnl.Layout.Row = 1; pnl.Layout.Column = 1;
    
    uilabel(pnl, 'Text', 'Ciudad:', 'Position', [20 350 200 20], 'FontWeight', 'bold');
    dd_city = uidropdown(pnl, 'Items', {Cities.Name}, 'Position', [20 320 240 30]);
    
    uilabel(pnl, 'Text', 'Hardware de Despliegue:', 'Position', [20 280 200 20], 'FontWeight', 'bold');
    dd_model = uidropdown(pnl, 'Items', {cfg.Models.Name}, 'Position', [20 250 240 30]);
    
    uilabel(pnl, 'Text', 'Frecuencia (MHz):', 'Position', [20 200 120 20]);
    ef_freq = uieditfield(pnl, 'numeric', 'Position', [160 200 100 25], 'Value', 915);
    
    uilabel(pnl, 'Text', 'Fondo de Validación 2D:', 'Position', [20 140 150 20], 'FontWeight', 'bold');
    dd_map = uidropdown(pnl, ...
        'Items', {'Topográfico (Relieve)', 'Calles (Urbano)', 'Satelital (Realismo)'}, ...
        'ItemsData', {'topographic', 'streets', 'satellite'}, ... 
        'Position', [20 110 240 30]);
    
    uibutton(pnl, 'Text', 'GENERAR GEMELO Y SIMULAR', 'Position', [20 30 240 50], ...
        'BackgroundColor', [0.15 0.45 0.25], 'FontColor', 'w', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(btn,event) run_morphological_sim(dd_city, dd_model, ef_freq, dd_map, Cities, cfg, ax_map, fig));
end

%% ========================================================================
%% NÚCLEO DE SIMULACIÓN FÍSICA
%% ========================================================================
function run_morphological_sim(dd_city, dd_model, ef_freq, dd_map, Cities, cfg, ax, fig)
    d = uiprogressdlg(fig, 'Title', 'Gemelo Morfológico', 'Message', 'Extrayendo datos OSM...');
    idx = strcmp({Cities.Name}, dd_city.Value); City = Cities(idx);
    
    T_Gemelo = Construir_Gemelo_Morfologico(City);
    if isempty(T_Gemelo), close(d); uialert(fig, 'Error OSM.', 'Error'); return; end
    
    d.Message = 'M-SAGA optimizando posiciones teóricas...';
    coords = [T_Gemelo.X_km, T_Gemelo.Y_km]; DistM = pdist2(coords, coords);
    cfg.radio_km = 1.0; BinM = DistM <= cfg.radio_km;
    [sol_msaga, ~] = Solver_MSAGA(BinM, cfg);
    idx_gws = find(sol_msaga);
    N_gws = length(idx_gws);
    
    d.Message = 'Calculando propagación y atenuación morfológica...';
    m_idx = strcmp({cfg.Models.Name}, dd_model.Value); HW = cfg.Models(m_idx);
    L0 = 20*log10(ef_freq.Value) - 27.55; 
    
    RSSI_Map = zeros(height(T_Gemelo), 1) - 150; 
    GW_Srv_Map = zeros(height(T_Gemelo), 1); % NUEVO: Memoria del Gateway servidor
    
    for i = 1:N_gws
        gw_pos = idx_gws(i);
        for nodo = 1:height(T_Gemelo)
            d_km = max(DistM(nodo, gw_pos), 0.01); d_m = d_km * 1000;
            n_path = 3.2; L_morf = 0;
            if strcmp(T_Gemelo.Zona(nodo), 'Parque'), L_morf = 8; 
            elseif strcmp(T_Gemelo.Zona(nodo), 'Densa'), L_morf = 18; n_path = 3.8; end
            
           % Sombra Topográfica (Modelo realista Sub-GHz / Difracción Knife-Edge)
            diff_alt = T_Gemelo.Alt_MSNM(gw_pos) - T_Gemelo.Alt_MSNM(nodo);
            if diff_alt > 15
                % El nodo está hundido (Valle). Atenuación logarítmica.
                L_topo = 15 * log10(diff_alt); 
            elseif diff_alt < -15
                % El nodo está elevado (Colina bloqueando el LoS hacia abajo).
                L_topo = 10 * log10(abs(diff_alt)); 
            else
                L_topo = 0; % En la misma meseta (Línea de Vista limpia)
            end
            
            % Límite físico empírico: La difracción del terreno en 915 MHz rara vez supera los 30 dB de castigo
            if L_topo > 30, L_topo = 30; end
            
            % Presupuesto de Enlace Final
            rssi = (HW.Ptx + HW.Gain) - L0 - 10*n_path*log10(d_m) - L_morf - HW.PenetrationLoss - L_topo;
            
            % Si este gateway ofrece mejor señal, lo registramos como servidor
            if rssi > RSSI_Map(nodo)
                RSSI_Map(nodo) = rssi;
                GW_Srv_Map(nodo) = i;
            end
        end
    end
    
    % ------------------------------------------------------------------------
    % PREPARACIÓN DE DATOS UNIVERSALES PARA TOOLTIPS (CONSISTENCIA)
    % ------------------------------------------------------------------------
    % 1. Coordenadas a formato DMS (Grados, Minutos, Segundos)
    dms_lat = arrayfun(@(v) convertir_DMS(v, true), T_Gemelo.Lat);
    dms_lon = arrayfun(@(v) convertir_DMS(v, false), T_Gemelo.Lon);
    
    % 2. Información cruzada y morfológica
    str_cruce = string(T_Gemelo.Calle_1) + " y " + string(T_Gemelo.Calle_2);
    str_zona  = string(T_Gemelo.Zona);
    str_alt   = string(T_Gemelo.Alt_MSNM) + " msnm";
    str_rssi  = string(round(RSSI_Map, 1)) + " dBm";
    
    % 3. Identidad del Gateway Servidor
    str_gw_srv = strings(height(T_Gemelo), 1);
    for k = 1:height(T_Gemelo)
        if GW_Srv_Map(k) == 0 || RSSI_Map(k) < HW.Sens
            str_gw_srv(k) = "Sin Servicio (Outage)";
        else
            str_gw_srv(k) = "Gateway " + string(GW_Srv_Map(k));
        end
    end

    idx_vivo = RSSI_Map >= HW.Sens;
    idx_muerto = RSSI_Map < HW.Sens;

    % Construcción de la Plantilla Universal de Filas para el Tooltip
    % ¡Al usar esto, 2D y 3D mostrarán EXACTAMENTE la misma información!
    rows_tooltip_muerto = [
        dataTipTextRow('Latitud: ', dms_lat(idx_muerto));
        dataTipTextRow('Longitud: ', dms_lon(idx_muerto));
        dataTipTextRow('Cruce: ', str_cruce(idx_muerto));
        dataTipTextRow('Morfología: ', str_zona(idx_muerto));
        dataTipTextRow('Altitud: ', str_alt(idx_muerto));
        dataTipTextRow('Servidor: ', str_gw_srv(idx_muerto));
        dataTipTextRow('Señal (Outage): ', str_rssi(idx_muerto))
    ];

    rows_tooltip_vivo = [
        dataTipTextRow('Latitud: ', dms_lat(idx_vivo));
        dataTipTextRow('Longitud: ', dms_lon(idx_vivo));
        dataTipTextRow('Cruce: ', str_cruce(idx_vivo));
        dataTipTextRow('Morfología: ', str_zona(idx_vivo));
        dataTipTextRow('Altitud: ', str_alt(idx_vivo));
        dataTipTextRow('Servidor: ', str_gw_srv(idx_vivo));
        dataTipTextRow('Potencia Rx: ', str_rssi(idx_vivo))
    ];

    rows_tooltip_gws = [
        dataTipTextRow('Latitud: ', dms_lat(idx_gws));
        dataTipTextRow('Longitud: ', dms_lon(idx_gws));
        dataTipTextRow('Identificador: ', "Gateway " + string(1:N_gws)');
        dataTipTextRow('Cruce: ', str_cruce(idx_gws));
        dataTipTextRow('Morfología: ', str_zona(idx_gws));
        dataTipTextRow('Altitud: ', str_alt(idx_gws))
    ];

    % ========================================================================
    % RENDERIZADO 2D (MAPA GEOGRÁFICO)
    % ========================================================================
    d.Message = 'Renderizando mapa 2D y 3D consistentes...';
    fig_map = figure('Name', sprintf('Validación Física 2D - %s', City.Name), 'Color', 'w', 'Position', [100 150 850 650]);
    gax = geoaxes(fig_map); geobasemap(gax, dd_map.Value); hold(gax, 'on');
    
    map_labels = {'Topográfico', 'Calles', 'Satelital'}; map_values = {'topographic', 'streets', 'satellite'};
    [~, initial_idx] = ismember(dd_map.Value, map_values); if initial_idx==0, initial_idx=1; end
    uicontrol(fig_map, 'Style', 'popupmenu', 'String', map_labels, 'Value', initial_idx, ...
        'Units', 'normalized', 'Position', [0.02 0.93 0.20 0.04], 'FontWeight', 'bold', ...
        'Callback', @(src, ~) geobasemap(gax, map_values{src.Value}));
    
    g_muerto = geoscatter(gax, T_Gemelo.Lat(idx_muerto), T_Gemelo.Lon(idx_muerto), 15, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.6);
    g_vivo = geoscatter(gax, T_Gemelo.Lat(idx_vivo), T_Gemelo.Lon(idx_vivo), 30, RSSI_Map(idx_vivo), 'filled', 'MarkerFaceAlpha', 0.85);
    sgw = geoscatter(gax, T_Gemelo.Lat(idx_gws), T_Gemelo.Lon(idx_gws), 300, 'w', '^', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    % INYECCIÓN DEL NÚMERO DE GATEWAY EN 2D
    text(gax, T_Gemelo.Lat(idx_gws), T_Gemelo.Lon(idx_gws), string(1:N_gws), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');

    % Asignar Plantillas 2D
    g_muerto.DataTipTemplate.DataTipRows = rows_tooltip_muerto;
    g_vivo.DataTipTemplate.DataTipRows = rows_tooltip_vivo;
    sgw.DataTipTemplate.DataTipRows = rows_tooltip_gws;
    
    colormap(gax, 'jet'); cb = colorbar(gax); cb.Label.String = 'Nivel de Señal RSSI (dBm)'; cb.Label.FontWeight = 'bold';
    clim(gax, [HW.Sens, -70]);
    title(gax, sprintf('Validación Física 2D: %s', HW.Name), 'FontSize', 14);

    % ========================================================================
    % RENDERIZADO 3D (CONSISTENCIA TOTAL)
    % ========================================================================
    fig3d = figure('Name', sprintf('Validación Física 3D - %s', City.Name), 'Color', 'w', 'Position', [960 150 850 650]);
    ax3d = axes(fig3d); hold(ax3d, 'on'); grid(ax3d, 'on');
    
    g3_muerto = scatter3(ax3d, T_Gemelo.Lon(idx_muerto), T_Gemelo.Lat(idx_muerto), T_Gemelo.Alt_MSNM(idx_muerto), 15, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);
    g3_vivo = scatter3(ax3d, T_Gemelo.Lon(idx_vivo), T_Gemelo.Lat(idx_vivo), T_Gemelo.Alt_MSNM(idx_vivo), 30, RSSI_Map(idx_vivo), 'filled', 'MarkerFaceAlpha', 0.95);
    sg3_gw = scatter3(ax3d, T_Gemelo.Lon(idx_gws), T_Gemelo.Lat(idx_gws), T_Gemelo.Alt_MSNM(idx_gws) + 5, 300, 'w', '^', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    % INYECCIÓN DEL NÚMERO DE GATEWAY EN 3D
    text(ax3d, T_Gemelo.Lon(idx_gws), T_Gemelo.Lat(idx_gws), T_Gemelo.Alt_MSNM(idx_gws) + 5, ...
         string(1:N_gws), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');

    % Asignar las MISMAS Plantillas 3D (Consistencia)
    g3_muerto.DataTipTemplate.DataTipRows = rows_tooltip_muerto;
    g3_vivo.DataTipTemplate.DataTipRows = rows_tooltip_vivo;
    sg3_gw.DataTipTemplate.DataTipRows = rows_tooltip_gws;
    
    % FORZAR EJES 3D A FORMATO DMS
    xt = ax3d.XTick; yt = ax3d.YTick;
    ax3d.XTickLabel = arrayfun(@(v) convertir_DMS(v, false), xt);
    ax3d.YTickLabel = arrayfun(@(v) convertir_DMS(v, true), yt);
    
    colormap(ax3d, 'jet'); clim(ax3d, [HW.Sens, -70]);
    cb3d = colorbar(ax3d); cb3d.Label.String = 'Nivel de Señal RSSI (dBm)'; cb3d.Label.FontWeight = 'bold';
    
    view(ax3d, -35, 35); 
    xlabel(ax3d, 'Longitud', 'FontWeight', 'bold'); ylabel(ax3d, 'Latitud', 'FontWeight', 'bold'); zlabel(ax3d, 'Altitud Topográfica (msnm)', 'FontWeight', 'bold');
    title(ax3d, sprintf('Validación Física 3D: %s', HW.Name), 'FontSize', 14);

    close(d);
    uialert(fig, 'Simulación 2D y 3D generadas exitosamente con consistencia total de datos.', 'Éxito');
end

%% ========================================================================
%% FUNCIONES AUXILIARES (DMS, GEMELO, M-SAGA)
%% ========================================================================
% NUEVA FUNCIÓN: Convierte decimales a Grados, Minutos, Segundos
function str = convertir_DMS(val, isLat)
    d = fix(val);
    rem = abs(val - d) * 60;
    m = fix(rem);
    s = (rem - m) * 60;
    if isLat
        if val >= 0, dir = 'N'; else, dir = 'S'; end
    else
        if val >= 0, dir = 'E'; else, dir = 'W'; end
    end
    str = string(sprintf('%d° %d'' %.1f" %c', abs(d), m, s, dir));
end

function T = Construir_Gemelo_Morfologico(City)
    dist = 0.015;
    q = sprintf('[out:json][timeout:45];(way["highway"~"^(primary|secondary|tertiary|residential)$"](%f,%f,%f,%f););out body;>;out skel qt;', ...
        City.Lat-dist, City.Lon-dist, City.Lat+dist, City.Lon+dist);
    try
        data = webread('https://overpass-api.de/api/interpreter', 'data', q, weboptions('Timeout', 45));
        c_nodes = containers.Map('KeyType','uint64','ValueType','any'); c_names = containers.Map('KeyType','uint64','ValueType','any');
        for i=1:length(data.elements)
            e = data.elements(i); if iscell(data.elements), e = data.elements{i}; end
            if strcmp(e.type,'node'), c_nodes(e.id) = [e.lat, e.lon];
            elseif strcmp(e.type,'way') && isfield(e, 'nodes')
                calle = "S/N"; if isfield(e, 'tags') && isfield(e.tags, 'name'), calle = string(e.tags.name); end
                for k=1:length(e.nodes)
                    nid = e.nodes(k); if iscell(e.nodes), nid = e.nodes{k}; end
                    if isKey(c_names, nid), c_names(nid) = [c_names(nid), calle]; else, c_names(nid) = calle; end
                end
            end
        end
        lat = []; lon = []; c1 = []; c2 = []; keys_n = c_names.keys;
        for k = 1:length(keys_n)
            nid = keys_n{k}; noms = c_names(nid);
            if length(noms) >= 2 && isKey(c_nodes, nid)
                coords = c_nodes(nid); lat = [lat; coords(1)]; lon = [lon; coords(2)]; c1 = [c1; noms(1)]; c2 = [c2; noms(2)];
            end
        end
        num_nodos = length(lat); R_earth = 6371; x_km = deg2rad(lon) * R_earth .* cosd(mean(lat)); y_km = deg2rad(lat) * R_earth;
        Alt_MSNM = round(City.Alt + 80 * sin(20 * x_km) .* cos(20 * y_km), 1);
        Zona = strings(num_nodos, 1);
        for i = 1:num_nodos
            r = rand();
            if abs(x_km(i) - mean(x_km)) < 0.3 && abs(y_km(i) - mean(y_km)) < 0.3, Zona(i) = "Densa";
            elseif r < 0.15, Zona(i) = "Parque"; else, Zona(i) = "Residencial"; end
        end
        T = unique(table(c1, c2, lat, lon, x_km, y_km, Alt_MSNM, Zona, 'VariableNames', {'Calle_1', 'Calle_2', 'Lat', 'Lon', 'X_km', 'Y_km', 'Alt_MSNM', 'Zona'}), 'rows');
    catch, T = []; end
end

function [sol, conv_gen] = Solver_MSAGA(BinM, cfg)
    [~, N] = size(BinM); Pop = rand(cfg.pop_size, N) < (cfg.target_gateways/N); Pop(1,:) = Solver_Greedy(BinM, cfg); 
    Fit = zeros(cfg.pop_size,1); conv_gen = 1; BestFit_Global = -inf;
    for g=1:cfg.max_iter
        for i=1:cfg.pop_size, Fit(i)=Fitness(Pop(i,:),BinM,cfg); end
        [curr_best_val, b] = max(Fit); Best = Pop(b,:);
        if curr_best_val > BestFit_Global, BestFit_Global = curr_best_val; conv_gen = g; end
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

function [sol, conv_gen] = Solver_Greedy(BinM, cfg)
    [~, N] = size(BinM); sol = false(1,N); cov = false(size(BinM,1),1); step = 0;
    for k=1:cfg.target_gateways
        best_g=-1; best_n=-1; cands=find(~sol);
        for i=cands, g=sum(BinM(:,i)&~cov); if g>best_g, best_g=g; best_n=i; end; end
        if best_n>0, sol(best_n)=true; cov=cov|BinM(:,best_n); step=step+1; else, break; end
    end
    conv_gen = step; 
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
            cands=find(~ind); if length(cands)>30, cands=cands(randperm(length(cands),30)); end
            bg=-1; bn=-1;
            for c=cands, g=sum(BinM(unc,c)); if g>bg, bg=g; bn=c; end, end
            if bn>0, ind(bn)=true; end
        end
    end
end