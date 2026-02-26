function ValidacionVisionOSM_V3_Fixed
    % =========================================================================
    % VALIDACIÓN FINAL V3: FIX ORIENTACIÓN Y VISUALIZACIÓN CLARA
    % Autor: Tesis MDPI
    % Cambios:
    % 1. Uso de 'roadmap' para ver claramente las calles.
    % 2. Fix 'flipud' para corregir mapa volteado.
    % 3. Zoom automático ajustado.
    % =========================================================================
    clc; close all; warning('off','all');
    
    % --- CONFIGURACIÓN ---
    apiKey = 'AIzaSyAnWExFoChjUJSrynNmAqKgNutZ9cL4Eq8'; 
    
    %% 1. CARGA DEL GEMELO DIGITAL
    files = dir('Gemelo_Digital_*.csv');
    if isempty(files), errordlg('No hay archivo Gemelo_Digital CSV.'); return; end
    [~, idx] = max([files.datenum]); 
    filename = files(idx).name;
    
    fprintf('📂 Cargando: %s\n', filename);
    TwinData = readtable(filename);
    
    % Filtro inteligente
    if ismember('Es_Interseccion', TwinData.Properties.VariableNames)
        Vision_Nodes = TwinData(TwinData.Es_Interseccion == 1, :);
    else
        Vision_Nodes = TwinData;
    end
    fprintf('   -> Nodos Visión: %d\n', height(Vision_Nodes));

    %% 2. DESCARGA DE DATOS REALES (OSM)
    fprintf('🌍 Descargando Ground Truth (OSM)...\n');
    % Margen de seguridad para la descarga
    pad = 0.003; 
    min_lat = min(TwinData.Lat) - pad; max_lat = max(TwinData.Lat) + pad;
    min_lon = min(TwinData.Lon) - pad; max_lon = max(TwinData.Lon) + pad;
    
    OSM_Table = Fetch_OSM_Intersections(min_lat, min_lon, max_lat, max_lon);
    
    if isempty(OSM_Table), errordlg('Error: No se pudieron descargar datos de OSM.'); return; end
    fprintf('   -> Intersecciones Reales encontradas: %d\n', height(OSM_Table));

    %% 3. CÁLCULO DE EFECTIVIDAD
    fprintf('⚙️  Calculando métricas de precisión...\n');
    TOLERANCIA_METROS = 25; 
    R_earth = 6371000; lat_ref = mean(Vision_Nodes.Lat);
    
    % Proyección rápida a metros para distancias euclidianas
    OSM_XY = [deg2rad(OSM_Table.Lon)*R_earth*cosd(lat_ref), deg2rad(OSM_Table.Lat)*R_earth];
    Vis_XY = [deg2rad(Vision_Nodes.Lon)*R_earth*cosd(lat_ref), deg2rad(Vision_Nodes.Lat)*R_earth];
    
    % Matching KD-Tree
    Mdl = createns(OSM_XY);
    [~, dist_match] = knnsearch(Mdl, Vis_XY);
    
    is_TP = dist_match <= TOLERANCIA_METROS; % Aciertos
    
    % Búsqueda inversa para Falsos Negativos
    Mdl_Vis = createns(Vis_XY);
    [~, dist_inv] = knnsearch(Mdl_Vis, OSM_XY);
    is_FN = dist_inv > TOLERANCIA_METROS;
    
    % Métricas Finales
    TP = sum(is_TP); FP = sum(~is_TP); FN = sum(is_FN);
    Precision = TP / (TP + FP + eps);
    Recall    = TP / (TP + FN + eps);
    F1_Score  = 2 * (Precision * Recall) / (Precision + Recall + eps);

    fprintf('\n📊 RESULTADOS FINALES:\n');
    fprintf('   Precision: %.2f%%\n   Recall:    %.2f%%\n   F1-Score:  %.2f%%\n', ...
        Precision*100, Recall*100, F1_Score*100);

    %% 4. GRAFICACIÓN CORREGIDA
    Graficar_Mapa_Fixed(OSM_Table, Vision_Nodes, is_TP, is_FN, ...
                        Precision, Recall, F1_Score, apiKey);
end

%% ========================================================================
%% FUNCIÓN DE GRAFICACIÓN (FIXED ORIENTATION & ROADMAP)
%% ========================================================================
function Graficar_Mapa_Fixed(T_OSM, T_Vis, is_TP, is_FN, prec, rec, f1, apiKey)
    
    f = figure('Name', 'Validación Intersecciones - Tesis MDPI', 'Color', 'w', ...
               'Units','normalized','Position',[0.05 0.05 0.9 0.85]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % --- PANEL 1: MAPA DE CALLES ---
    ax1 = nexttile;
    
    % 1. Calcular Centro y Zoom dinámico
    all_lat = [T_OSM.Lat; T_Vis.Lat];
    all_lon = [T_OSM.Lon; T_Vis.Lon];
    c_lat = mean(all_lat); c_lon = mean(all_lon);
    
    % Cálculo de Zoom basado en la extensión de los datos
    span_lat = max(all_lat) - min(all_lat);
    zoom = floor(log2(360/span_lat)) + 1; 
    zoom = min(max(zoom, 15), 19); % Zoom alto para ver calles (15-19)
    
    fprintf('🖼️  Descargando Mapa de Calles (Roadmap, Zoom %d)...\n', zoom);
    
    sz = 640;
    % Estilo: Simplificado (sin etiquetas de texto) para ver mejor los puntos
    style = '&style=feature:all|element:labels|visibility:off'; 
    
    url = sprintf('https://maps.googleapis.com/maps/api/staticmap?center=%f,%f&zoom=%d&size=%dx%d&maptype=roadmap&key=%s%s', ...
        c_lat, c_lon, zoom, sz, sz, apiKey, style);
    
    try
        img = webread(url);
        
        % --- FIX IMPORTANTE: INVERSIÓN DE IMAGEN ---
        % Las imágenes vienen indexadas [0,0] arriba-izquierda.
        % Los plots geográficos tienen el [0,0] abajo-izquierda.
        % Solución: Volteamos la imagen verticalmente.
        img = flipud(img); 
        
        % Referencia Geoespacial
        nTiles = 2^zoom;
        meters_per_pixel = 156543.03392 * cosd(c_lat) / nTiles;
        deg_lat_pp = (meters_per_pixel/1000)/111.32;
        deg_lon_pp = (meters_per_pixel/1000)/(111.32*cosd(c_lat));
        
        % Límites (Lat ascendente: Sur -> Norte)
        lat_lim = [c_lat - (sz/2)*deg_lat_pp, c_lat + (sz/2)*deg_lat_pp];
        lon_lim = [c_lon - (sz/2)*deg_lon_pp, c_lon + (sz/2)*deg_lon_pp];
        
        R = imref2d(size(img), lon_lim, lat_lim);
        
        imshow(img, R, 'Parent', ax1); 
        set(ax1, 'YDir', 'normal'); % Asegura que el eje Y crezca hacia arriba
        hold(ax1, 'on');
        
    catch ME
        warning('Error cargando mapa: %s. Usando fondo blanco.', ME.message);
        axis(ax1, 'equal'); hold(ax1, 'on');
    end
    
    % --- PLOTEO DE PUNTOS ---
    % 1. Ground Truth (OSM) - Círculos negros huecos
    scatter(ax1, T_OSM.Lon, T_OSM.Lat, 60, 'k', 'LineWidth', 1.5, ...
        'DisplayName', 'Realidad (OSM)');
    
    % 2. Falsos Negativos (No detectados) - Cuadros Azules
    % Importante: Ploteamos esto antes para que los aciertos queden encima
    scatter(ax1, T_OSM.Lon(is_FN), T_OSM.Lat(is_FN), 50, 's', 'filled', ...
        'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', 'w', ...
        'DisplayName', 'No Detectado (FN)');
    
    % 3. Falsos Positivos (Ruido) - Cruces Rojas
    scatter(ax1, T_Vis.Lon(~is_TP), T_Vis.Lat(~is_TP), 40, 'x', ...
        'MarkerEdgeColor', '#D95319', 'LineWidth', 2, ...
        'DisplayName', 'Falso Positivo (FP)');
        
    % 4. Aciertos (TP) - Puntos Verdes
    scatter(ax1, T_Vis.Lon(is_TP), T_Vis.Lat(is_TP), 30, 'o', 'filled', ...
        'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Acierto (TP)');
    
    title(ax1, 'Validación Visual de Intersecciones');
    xlabel(ax1, 'Longitud'); ylabel(ax1, 'Latitud');
    legend(ax1, 'Location', 'southoutside', 'Orientation', 'horizontal');
    box(ax1, 'on'); axis(ax1, 'tight');
    
    % --- PANEL 2: ESTADÍSTICAS ---
    ax2 = nexttile;
    vals = [prec, rec, f1] * 100;
    
    b = bar(ax2, vals);
    b.FaceColor = 'flat';
    b.CData(1,:) = [0.4660 0.6740 0.1880]; % Verde
    b.CData(2,:) = [0 0.4470 0.7410];      % Azul
    b.CData(3,:) = [0.9290 0.6940 0.1250]; % Amarillo
    
    xticklabels(ax2, {'Precision', 'Recall', 'F1-Score'});
    ylim(ax2, [0 115]);
    ylabel(ax2, 'Porcentaje (%)');
    title(ax2, sprintf('Métricas de Rendimiento\nF1-Score: %.1f%%', f1*100));
    grid(ax2, 'on');
    
    text(ax2, 1:3, vals, num2str(vals', '%.1f%%'), ...
        'vert','bottom','horiz','center', 'FontSize', 12, 'FontWeight', 'bold');
        
    % Cuadro de Explicación
    dim = [0.55 0.2 0.35 0.2];
    str = {'INTERPRETACIÓN VISUAL:', ...
           '🟢 Verde: Coincidencia Perfecta (Intersección Real detectada).', ...
           '❌ Rojo: Algoritmo vio calle donde no había (Posible garaje/patio).', ...
           '🟦 Azul: Intersección real que el algoritmo no vio (Oclusión).', ...
           '⚫ Círculo Negro: Referencia de dónde está la esquina real.'};
    annotation('textbox', dim, 'String', str, 'FitBoxToText','on', 'BackgroundColor', 'w');
end

%% ========================================================================
%% FUNCIÓN AUXILIAR DE DESCARGA (ROBUSTA)
%% ========================================================================
function OSM_Table = Fetch_OSM_Intersections(min_lat, min_lon, max_lat, max_lon)
    api_url = 'https://overpass-api.de/api/interpreter';
    
    % Query optimizada para intersecciones viales
    osm_query = sprintf(['[out:json][timeout:60];' ...
        '(' ...
        'way["highway"~"^(primary|secondary|tertiary|residential|unclassified)$"](%f,%f,%f,%f);' ...
        ');' ...
        'out body;>;out skel qt;'], min_lat, min_lon, max_lat, max_lon);
    
    try
        options = weboptions('Timeout', 60, 'ContentType', 'json');
        data = webwrite(api_url, 'data', osm_query, options);
    catch
        OSM_Table = []; return;
    end
    
    if ~isfield(data, 'elements') || isempty(data.elements), OSM_Table = []; return; end
    
    raw = data.elements;
    node_coords = containers.Map('KeyType','double','ValueType','any');
    node_counts = containers.Map('KeyType','double','ValueType','double');
    
    % 1. Mapeo Nodos
    for i = 1:length(raw)
        if iscell(raw), el = raw{i}; else, el = raw(i); end
        if strcmp(el.type, 'node'), node_coords(el.id) = [el.lat, el.lon]; end
    end
    
    % 2. Conteo Vías
    for i = 1:length(raw)
        if iscell(raw), el = raw{i}; else, el = raw(i); end
        if strcmp(el.type, 'way') && isfield(el, 'nodes') && ~isempty(el.nodes)
            u_nodes = unique(el.nodes);
            for k=1:length(u_nodes)
                nid = u_nodes(k);
                if isKey(node_counts, nid), node_counts(nid)=node_counts(nid)+1; else, node_counts(nid)=1; end
            end
        end
    end
    
    % 3. Tabla
    keys = cell2mat(node_counts.keys()); vals = cell2mat(node_counts.values());
    inter_ids = keys(vals >= 2);
    lats = []; lons = [];
    for i=1:length(inter_ids)
        if isKey(node_coords, inter_ids(i))
            c = node_coords(inter_ids(i));
            lats = [lats; c(1)]; lons = [lons; c(2)];
        end
    end
    OSM_Table = table(lats, lons, 'VariableNames', {'Lat','Lon'});
end