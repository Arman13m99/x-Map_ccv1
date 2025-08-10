// script.js (Complete updated file with improved heatmap system)
document.addEventListener('DOMContentLoaded', () => {
    let map;
    let vendorLayerGroup = L.featureGroup();
    let polygonLayerGroup = L.featureGroup();
    let coverageGridLayerGroup = L.featureGroup();
    let heatmapLayer;
    let tempLocationMarker = null; // Variable for the find-on-map marker
    let currentHeatmapType = 'none';
    let showVendorRadius = true;
    let vendorsAreVisible = true;
    let allVendorsData = [];
    let allPolygonsData = null;
    let allCoverageGridData = [];
    let initialFilterData = {};
    let lastHeatmapData = null;
    let currentRadiusModifier = 1.0;
    let currentRadiusMode = 'percentage';
    let currentRadiusFixed = 3.0;
    let marketingAreasOnTop = false;
    
    // NEW: Improved heatmap management variables
    let currentZoomLevel = 11;
    let heatmapConfig = {
        autoOptimize: true,
        baseRadius: 25,
        baseBlur: 15,
        maxIntensity: 1.0,
        smoothTransitions: true,
        zoomIndependent: true
    };
    let lastOptimalParams = null;

    const API_BASE_URL = '/api';
    
    // --- DOM Elements ---
    const bodyEl = document.body;
    
    // Global debug function
    window.debugVendors = function() {
        console.log("=== VENDOR DEBUG ===");
        console.log("vendorsAreVisible:", vendorsAreVisible);
        console.log("allVendorsData length:", allVendorsData ? allVendorsData.length : "undefined");
        console.log("vendorLayerGroup layers:", vendorLayerGroup ? vendorLayerGroup.getLayers().length : "undefined");
        console.log("vendorLayerGroup on map:", map && vendorLayerGroup ? map.hasLayer(vendorLayerGroup) : "undefined");
        console.log("map center:", map ? map.getCenter() : "undefined");
        console.log("map zoom:", map ? map.getZoom() : "undefined");
        
        // Try to force a redraw
        console.log("Forcing vendor redraw...");
        redrawVendorMarkersAndRadii();
        
        return {
            vendorsVisible: vendorsAreVisible,
            vendorCount: allVendorsData ? allVendorsData.length : 0,
            layersInGroup: vendorLayerGroup ? vendorLayerGroup.getLayers().length : 0,
            layerOnMap: map && vendorLayerGroup ? map.hasLayer(vendorLayerGroup) : false
        };
    };
    
    // Global function to test API directly with no filters
    window.testDirectAPI = function() {
        console.log("Testing API directly with minimal parameters...");
        
        fetch('/api/map-data?city=tehran')
            .then(response => response.json())
            .then(data => {
                console.log("Direct API test result:");
                console.log("- Vendors received:", data.vendors?.length || 0);
                if (data.vendors && data.vendors.length > 0) {
                    console.log("- Sample vendor:", data.vendors[0]);
                    
                    // Update the frontend data directly
                    allVendorsData = data.vendors;
                    console.log("- Updated allVendorsData, calling updateMapLayers...");
                    updateMapLayers();
                }
            })
            .catch(error => console.error("Direct API test failed:", error));
    };
    
    // Optional heavy reset function for debugging purposes only (not used in normal operation)
    window.resetAllFilters = function() {
        console.warn("resetAllFilters() is deprecated and causes heavy load. Use with caution.");
        
        // Clear ALL checkboxes on the entire page
        document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.checked = false;
        });
        
        // Reset all select elements to their first option
        document.querySelectorAll('select').forEach(select => {
            select.selectedIndex = 0;
        });
        
        // Clear text inputs (except coordinate finders)
        document.querySelectorAll('input[type="text"], textarea').forEach(input => {
            if (input.id !== 'lat-finder-input' && input.id !== 'lng-finder-input') {
                input.value = "";
            }
        });
        
        // Update custom dropdown button texts
        Object.values(customFilterConfigs).forEach(config => {
            updateCustomDropdownButtonText(config);
        });
        
        console.log("All filters reset (heavy operation). Fetching fresh data...");
        fetchAndDisplayMapData();
    };
    const daterangeStartEl = document.getElementById('daterange-start');
    const daterangeEndEl = document.getElementById('daterange-end');
    const cityEl = document.getElementById('city');
    const areaMainTypeEl = document.getElementById('area-main-type');
    const vendorCodesFilterEl = document.getElementById('vendor-codes-filter');
    const vendorVisibleEl = document.getElementById('vendor-visible');
    const vendorIsOpenEl = document.getElementById('vendor-is-open');
    const vendorRadiusToggleBtn = document.getElementById('vendor-radius-toggle');
    const radiusEdgeColorEl = document.getElementById('radius-edge-color');
    const radiusInnerColorEl = document.getElementById('radius-inner-color');
    const radiusInnerNoneEl = document.getElementById('radius-inner-none');
    const vendorMarkerSizeEl = document.getElementById('vendor-marker-size');
    const markerSizeValueEl = document.getElementById('marker-size-value');
    const areaFillColorEl = document.getElementById('area-fill-color');
    const areaFillNoneEl = document.getElementById('area-fill-none');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');
    
    // Radius modifier elements
    const vendorRadiusModifierEl = document.getElementById('vendor-radius-modifier');
    const radiusModifierValueEl = document.getElementById('radius-modifier-value');
    const btnResetRadius = document.getElementById('btn-reset-radius');
    
    // Radius mode elements
    const radiusModeSelector = document.getElementById('radius-mode-selector');
    const radiusPercentageControl = document.getElementById('radius-percentage-control');
    const radiusFixedControl = document.getElementById('radius-fixed-control');
    const vendorRadiusFixedEl = document.getElementById('vendor-radius-fixed');
    const radiusFixedValueEl = document.getElementById('radius-fixed-value');
    
    // Grid visualization elements
    const gridBlurEl = document.getElementById('grid-blur');
    const gridBlurValueEl = document.getElementById('grid-blur-value');
    const gridFadeEl = document.getElementById('grid-fade');
    const gridFadeValueEl = document.getElementById('grid-fade-value');
    const marketingAreasOnTopBtn = document.getElementById('marketing-areas-on-top');
    const gridVisualizationSection = document.getElementById('grid-visualization-section');
    const gridPointSizeEl = document.getElementById('grid-point-size');
    const gridPointSizeValueEl = document.getElementById('grid-point-size-value');
    
    // Lat/Lng finder elements
    const latFinderInputEl = document.getElementById('lat-finder-input');
    const lngFinderInputEl = document.getElementById('lng-finder-input');
    const btnFindLocation = document.getElementById('btn-find-location');
    
    let globalLoadingOverlayEl = document.getElementById('map-loading-overlay-wrapper');
    if (!globalLoadingOverlayEl) {
        globalLoadingOverlayEl = document.createElement('div');
        globalLoadingOverlayEl.id = 'map-loading-overlay-wrapper';
        document.body.appendChild(globalLoadingOverlayEl); 
    }
    
    const mapTypeButtons = {
        densityTotal: document.getElementById('btn-order-density-total'),
        densityOrganic: document.getElementById('btn-order-density-organic'),
        densityNonOrganic: document.getElementById('btn-order-density-non-organic'),
        userDensity: document.getElementById('btn-user-density-heatmap'),
        population: document.getElementById('btn-population-heatmap'),
        vendors: document.getElementById('btn-vendors-map'),
    };
    const btnToggleVendors = document.getElementById('btn-toggle-vendors');
    const btnClearHeatmap = document.getElementById('btn-clear-heatmap');
    
    // Heatmap control elements
    const heatmapRadiusEl = document.getElementById('heatmap-radius');
    const heatmapRadiusValueEl = document.getElementById('heatmap-radius-value');
    const heatmapBlurEl = document.getElementById('heatmap-blur');
    const heatmapBlurValueEl = document.getElementById('heatmap-blur-value');
    const heatmapMaxValEl = document.getElementById('heatmap-max-val');
    const heatmapMaxValValueEl = document.getElementById('heatmap-max-val-value');

    const customFilterConfigs = {
        areaSubType: {
            button: document.getElementById('area-sub-type-filter-button'),
            panel: document.getElementById('area-sub-type-filter-panel'),
            paramName: 'area_sub_type_filter', defaultText: 'Select Sub Types', optionsData: []
        },
        businessLine: {
            button: document.getElementById('business-line-filter-button'),
            panel: document.getElementById('business-line-filter-panel'),
            paramName: 'business_lines', defaultText: 'Select Business Lines', optionsData: []
        },
        vendorStatus: { 
            button: document.getElementById('vendor-status-filter-button'),
            panel: document.getElementById('vendor-status-filter-panel'),
            paramName: 'vendor_status_ids', defaultText: 'Select Statuses', optionsData: []
        },
        vendorGrade: { 
            button: document.getElementById('vendor-grade-filter-button'),
            panel: document.getElementById('vendor-grade-filter-panel'),
            paramName: 'vendor_grades', defaultText: 'Select Grades', optionsData: []
        },
    };

    let defaultVendorIcon = L.icon({
        iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
        iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        iconSize: [12 * 2, 12 * (41/25) * 2],
        iconAnchor: [12, 12 * (41/25) * 2],
        popupAnchor: [0, -12 * (41/25) * 2],
        shadowSize: [12 * (41/25) * 2, 12 * (41/25) * 2],
    });

    // NEW: Enhanced heatmap configuration functions
    function calculateOptimalHeatmapParams(data, zoomLevel) {
        if (!data || data.length === 0) {
            return { max: 1.0, radius: 25, blur: 15 };
        }

        const values = data.map(point => point.value).filter(v => v != null && !isNaN(v));
        if (values.length === 0) {
            return { max: 1.0, radius: 25, blur: 15 };
        }

        // Calculate percentiles for robust statistics
        values.sort((a, b) => a - b);
        const p75 = values[Math.floor(values.length * 0.75)];
        const p90 = values[Math.floor(values.length * 0.90)];
        const p95 = values[Math.floor(values.length * 0.95)];

        // Set max to 90th percentile to avoid outlier dominance
        let optimalMax = p90 / 100.0;
        if (optimalMax <= 0) optimalMax = 1.0;
        
        // Zoom-dependent radius scaling
        let radiusMultiplier = 1.0;
        if (zoomLevel <= 10) {
            radiusMultiplier = 1.8;
        } else if (zoomLevel <= 12) {
            radiusMultiplier = 1.4;
        } else if (zoomLevel >= 16) {
            radiusMultiplier = 0.7;
        } else if (zoomLevel >= 14) {
            radiusMultiplier = 0.85;
        }

        const optimalRadius = Math.round(heatmapConfig.baseRadius * radiusMultiplier);

        // Data density-dependent blur
        const dataDensity = data.length / 1000;
        let blurMultiplier = 1.0;
        if (dataDensity > 3) {
            blurMultiplier = 1.4;  // More blur for very dense data
        } else if (dataDensity > 1.5) {
            blurMultiplier = 1.2;  // Slightly more blur for dense data
        } else if (dataDensity < 0.5) {
            blurMultiplier = 0.8;  // Less blur for sparse data
        }

        const optimalBlur = Math.round(heatmapConfig.baseBlur * blurMultiplier);

        return {
            max: Math.max(0.1, Math.min(3.0, optimalMax)),
            radius: Math.max(5, Math.min(60, optimalRadius)),
            blur: Math.max(5, Math.min(40, optimalBlur))
        };
    }

    function getZoomAdjustedHeatmapOptions() {
        const currentZoom = map.getZoom();
        const userRadius = parseInt(heatmapRadiusEl.value);
        const userBlur = parseInt(heatmapBlurEl.value);
        const userMax = parseFloat(heatmapMaxValEl.value);
        
        if (!heatmapConfig.autoOptimize) {
            // Use user-defined values with minimal zoom adjustment
            const zoomFactor = Math.pow(1.1, currentZoom - 11); // Reference zoom 11
            return {
                radius: Math.round(userRadius * Math.min(2, Math.max(0.5, zoomFactor))),
                blur: userBlur,
                max: userMax,
                minOpacity: 0.3
            };
        }

        // Auto-optimize mode
        if (lastOptimalParams && heatmapConfig.smoothTransitions) {
            // Smooth transition to new optimal parameters
            const optimal = calculateOptimalHeatmapParams(lastHeatmapData, currentZoom);
            return {
                radius: Math.round((lastOptimalParams.radius + optimal.radius) / 2),
                blur: Math.round((lastOptimalParams.blur + optimal.blur) / 2),
                max: (lastOptimalParams.max + optimal.max) / 2,
                minOpacity: 0.3
            };
        }

        return calculateOptimalHeatmapParams(lastHeatmapData, currentZoom);
    }

    function updateHeatmapControlsDisplay(optimalParams) {
        if (!heatmapConfig.autoOptimize || !optimalParams) return;

        // Update sliders to show optimal values without triggering events
        const originalRadiusHandler = heatmapRadiusEl.oninput;
        const originalBlurHandler = heatmapBlurEl.oninput;
        const originalMaxHandler = heatmapMaxValEl.oninput;

        heatmapRadiusEl.oninput = null;
        heatmapBlurEl.oninput = null;
        heatmapMaxValEl.oninput = null;

        heatmapRadiusEl.value = optimalParams.radius;
        heatmapRadiusValueEl.textContent = optimalParams.radius;

        heatmapBlurEl.value = optimalParams.blur;
        heatmapBlurValueEl.textContent = optimalParams.blur;

        heatmapMaxValEl.value = optimalParams.max.toFixed(1);
        heatmapMaxValValueEl.textContent = optimalParams.max.toFixed(1);

        // Restore event handlers
        setTimeout(() => {
            heatmapRadiusEl.oninput = originalRadiusHandler;
            heatmapBlurEl.oninput = originalBlurHandler;
            heatmapMaxValEl.oninput = originalMaxHandler;
        }, 100);
    }

    function updateVendorIconSize(baseSize) { 
        const aspectRatio = 41/25;
        const iconWidth = parseInt(baseSize);
        const iconHeight = parseInt(iconWidth * aspectRatio);
        defaultVendorIcon = L.icon({
            iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
            iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize: [iconWidth, iconHeight],
            iconAnchor: [iconWidth / 2, iconHeight],
            popupAnchor: [0, -iconHeight + 5],
            shadowSize: [iconWidth * 1.5, iconHeight * 1.5],
            shadowAnchor: [iconWidth / 3, iconHeight*1.4]
        });
        markerSizeValueEl.textContent = baseSize;
        if(vendorsAreVisible) redrawVendorMarkersAndRadii();
    }

    function init() {
        initMap();
        fetchInitialFilterData().then(() => {
            populateCitySelect();
            initializeCustomDropdowns();
            applyDefaultFilters(); 
            setupEventListeners();
            setupHeatmapZoomHandler(); // NEW: Setup zoom handler

            const today = new Date(); 
            const thirtyDaysAgo = new Date(new Date().setDate(today.getDate() - 30));
            daterangeStartEl.value = thirtyDaysAgo.toISOString().split('T')[0];
            daterangeEndEl.value = today.toISOString().split('T')[0];
            updateVendorIconSize(vendorMarkerSizeEl.value); 
            
            // Enhanced default heatmap values
            heatmapRadiusEl.value = "25";
            heatmapRadiusValueEl.textContent = "25";
            heatmapBlurEl.value = "15";
            heatmapBlurValueEl.textContent = "15";
            heatmapMaxValEl.value = "1.0";
            heatmapMaxValValueEl.textContent = "1.0";
            
            // Enhanced tooltips
            heatmapRadiusEl.title = "Heatmap point spread radius. Auto-adjusts with zoom when optimized.";
            heatmapBlurEl.title = "Smoothness of heat transitions. Auto-adjusts based on data density.";
            heatmapMaxValEl.title = "Maximum intensity threshold. Auto-optimized to data distribution.";
            
            // Initialize radius modifier
            vendorRadiusModifierEl.value = "100";
            radiusModifierValueEl.textContent = "100";
            currentRadiusModifier = 1.0;
            currentRadiusMode = 'percentage';
            currentRadiusFixed = 3.0;
            vendorRadiusFixedEl.value = "3";
            radiusFixedValueEl.textContent = "3";
            
            // Initialize grid visualization controls
            gridBlurEl.value = "0";
            gridBlurValueEl.textContent = "0";
            gridFadeEl.value = "100";
            gridFadeValueEl.textContent = "100";
            gridPointSizeEl.value = "6";
            gridPointSizeValueEl.textContent = "6";
            marketingAreasOnTop = false;
            marketingAreasOnTopBtn.classList.remove('active');
            // Grid visualization section is now always visible
            
            btnToggleVendors.textContent = vendorsAreVisible ? 'Vendors On' : 'Vendors Off';
            btnToggleVendors.classList.toggle('active', vendorsAreVisible);
            fetchAndDisplayMapData();
        }).catch(error => {
            console.error("Initialization failed:", error);
            showLoading(true, `Initialization failed: ${error.message}. Please refresh.`);
        });
    }

    function initMap() {
        const mapContainer = document.getElementById('map');
        if(mapContainer) mapContainer.innerHTML = ''; else { console.error("Map container ('map') not found!"); return; }
        map = L.map('map', { 
            preferCanvas: true,
            attributionControl: false 
        }).setView([35.7219, 51.3347], 11);
        // Create dedicated panes to control layer order and clickability
        map.createPane('polygonPane');
        map.getPane('polygonPane').style.zIndex = 450; 
        
        map.createPane('coverageGridPane');
        map.getPane('coverageGridPane').style.zIndex = 460;
        
        // Use the default shadow pane for radii to ensure they are behind polygons
        map.getPane('shadowPane').style.zIndex = 250;
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd', maxZoom: 20
        }).addTo(map);
        L.control.attribution({position: 'bottomleft'}).addTo(map);
        
        if (vendorsAreVisible) vendorLayerGroup.addTo(map);
        polygonLayerGroup.addTo(map);
        coverageGridLayerGroup.addTo(map);
    }

    // NEW: Zoom event handler for consistent heatmap rendering
    function setupHeatmapZoomHandler() {
        let zoomTimeout;
        
        map.on('zoomstart', () => {
            // Clear any pending zoom updates
            if (zoomTimeout) {
                clearTimeout(zoomTimeout);
            }
        });

        map.on('zoomend', () => {
            const newZoomLevel = map.getZoom();
            
            // Only update if zoom level actually changed significantly
            if (Math.abs(newZoomLevel - currentZoomLevel) >= 0.5) {
                currentZoomLevel = newZoomLevel;
                
                // Debounce the heatmap update
                if (zoomTimeout) {
                    clearTimeout(zoomTimeout);
                }
                
                zoomTimeout = setTimeout(() => {
                    if (currentHeatmapType !== 'none' && lastHeatmapData) {
                        renderCurrentHeatmap();
                    }
                }, 200); // Small delay to ensure smooth zooming
            }
        });
    }

    async function fetchInitialFilterData() {
        showLoading(true, 'Fetching initial settings...');
        try {
            const response = await fetch(`${API_BASE_URL}/initial-data`);
            if (!response.ok) {
                const errorText = await response.text();
                showLoading(true, `Error fetching settings: ${response.status}. Please refresh.`);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            initialFilterData = await response.json();
            if (initialFilterData.error) {
                showLoading(true, `Backend error (settings): ${initialFilterData.error}. Please refresh.`);
                 throw new Error(`Backend error (initial data): ${initialFilterData.error} - ${initialFilterData.details || ''}`);
            }
        } catch (error) {
            console.error("Failed to fetch initial filter data:", error);
            showLoading(true, `Failed to load settings: ${error.message}. Please refresh.`);
            throw error;
        }
    }
    
    function populateCitySelect() {
        if (!initialFilterData || !initialFilterData.cities) return;
        populateSelectWithOptions(cityEl, initialFilterData.cities.map(c => ({ value: c.name, text: c.name })), false);
        cityEl.value = "tehran";
    }

    function initializeCustomDropdowns() {
        if (!initialFilterData) return;
        customFilterConfigs.businessLine.optionsData = (initialFilterData.business_lines || [])
            .map(bl => ({ value: bl, text: bl, checked: false }));
        renderCustomDropdown(customFilterConfigs.businessLine);
        
        customFilterConfigs.vendorStatus.optionsData = (initialFilterData.vendor_statuses || [])
            .map(s => ({ value: String(s), text: `Status ${s}`, checked: false }));
        renderCustomDropdown(customFilterConfigs.vendorStatus);
        customFilterConfigs.vendorGrade.optionsData = (initialFilterData.vendor_grades || [])
            .map(g => ({ value: g, text: g, checked: false }));
        renderCustomDropdown(customFilterConfigs.vendorGrade);
        updateCityDependentCustomFilters(cityEl.value);
    }

    function applyDefaultFilters() {
        // Apply specific default filter values for initial load/refresh
        console.log("Applying specific default filter values...");
        
        // Set specific default values
        vendorVisibleEl.value = "1";  // Yes (visible vendors only)
        vendorIsOpenEl.value = "any"; // Any (open/closed doesn't matter)
        
        // Set business_line = Restaurant
        const blConfig = customFilterConfigs.businessLine;
        if (blConfig && blConfig.checkboxes) {
            // First uncheck all business lines
            blConfig.checkboxes.forEach(cb => cb.checked = false);
            // Check only Restaurant
            const restaurantCheckbox = blConfig.checkboxes.find(cb => cb.value === 'Restaurant');
            if (restaurantCheckbox) {
                restaurantCheckbox.checked = true;
            }
        }
        
        // Set vendor_status_id = 5
        const statusConfig = customFilterConfigs.vendorStatus;
        if (statusConfig && statusConfig.checkboxes) {
            // First uncheck all statuses
            statusConfig.checkboxes.forEach(cb => cb.checked = false);
            // Check only status 5
            const status5Checkbox = statusConfig.checkboxes.find(cb => cb.value === '5');
            if (status5Checkbox) {
                status5Checkbox.checked = true;
            }
        }
        
        // Set vendor_grade = ALL (leave all grades unchecked to show all)
        const gradeConfig = customFilterConfigs.vendorGrade;
        if (gradeConfig && gradeConfig.checkboxes) {
            // Uncheck all grades to show ALL grades
            gradeConfig.checkboxes.forEach(cb => cb.checked = false);
        }
        
        // Update dropdown button texts to reflect the changes
        Object.values(customFilterConfigs).forEach(config => {
            updateCustomDropdownButtonText(config);
        });
        
        console.log("Default filter values applied: Restaurant, visible=1, open=any, status=5, grade=all");
    }
    
    function updateCityDependentCustomFilters(selectedCity) {
        updateAreaSubTypeCustomFilter();
    }

    function updateAreaSubTypeCustomFilter() {
        if (!initialFilterData) return;
        const selectedAreaMainType = areaMainTypeEl.value;
        const selectedCity = cityEl.value;
        let subAreaOptionObjects = [];
        
        if (selectedAreaMainType === "coverage_grid") {
            // For coverage grid, sub-types should filter the polygons *on top*, not the grid itself
            if (initialFilterData.marketing_areas_by_city && initialFilterData.marketing_areas_by_city[selectedCity]) {
                subAreaOptionObjects = initialFilterData.marketing_areas_by_city[selectedCity].map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            }
        } else if (selectedAreaMainType === "tapsifood_marketing_areas") {
            if (initialFilterData.marketing_areas_by_city && initialFilterData.marketing_areas_by_city[selectedCity]) {
                subAreaOptionObjects = initialFilterData.marketing_areas_by_city[selectedCity].map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            }
        } else if (selectedCity === "tehran") {
            if (selectedAreaMainType === "tehran_region_districts" && initialFilterData.tehran_region_districts) {
                subAreaOptionObjects = initialFilterData.tehran_region_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            } else if (selectedAreaMainType === "tehran_main_districts" && initialFilterData.tehran_main_districts) {
                subAreaOptionObjects = initialFilterData.tehran_main_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            } else if (selectedAreaMainType === "all_tehran_districts") {
                const regionD = (initialFilterData.tehran_region_districts || []).map(name => ({value: name, text: `Region: ${decodeURIComponentSafe(name)}`}));
                const mainD = (initialFilterData.tehran_main_districts || []).map(name => ({value: name, text: `Main: ${decodeURIComponentSafe(name)}`}));
                subAreaOptionObjects = [...regionD, ...mainD];
            }
        }
        customFilterConfigs.areaSubType.optionsData = subAreaOptionObjects
            .map(opt => ({ value: opt.value, text: opt.text, checked: false }));
        renderCustomDropdown(customFilterConfigs.areaSubType);
    }
    

    function decodeURIComponentSafe(str) {
        if (!str) return str;
        try {
            return decodeURIComponent(str.replace(/\+/g, ' ')); 
        } catch (e) {
            console.warn(`Failed to decode URI component: ${str}`, e);
            return str;
        }
    }
    
    function populateSelectWithOptions(selectElement, options, isMultiple = true) {
        selectElement.innerHTML = '';
        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.text;
            selectElement.add(option);
        });
    }

    function renderCustomDropdown(config) {
        config.panel.innerHTML = '';
        if (!config.optionsData || config.optionsData.length === 0) {
            config.panel.innerHTML = '<div class="dropdown-panel-item" style="color:var(--text-muted); cursor:default;">No options available</div>';
            updateCustomDropdownButtonText(config);
            return;
        }
        config.optionsData.forEach((option, index) => {
            const itemDiv = document.createElement('div');
            itemDiv.classList.add('dropdown-panel-item');
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `${config.paramName}-opt-${index}-${Math.random().toString(36).substr(2, 5)}`;
            checkbox.value = option.value;
            checkbox.checked = option.checked;
            const label = document.createElement('label');
            label.htmlFor = checkbox.id;
            label.textContent = option.text;
            checkbox.addEventListener('change', () => {
                option.checked = checkbox.checked;
                updateCustomDropdownButtonText(config);
            });
            itemDiv.addEventListener('click', (e) => {
                 if (e.target !== checkbox) checkbox.click();
            });
            itemDiv.appendChild(checkbox);
            itemDiv.appendChild(label);
            config.panel.appendChild(itemDiv);
        });
        updateCustomDropdownButtonText(config);
    }

    function updateCustomDropdownButtonText(config) {
        const selectedOptions = config.optionsData.filter(opt => opt.checked);
        const selectedCount = selectedOptions.length;
        if (selectedCount === 0) {
            config.button.textContent = config.defaultText;
        } else if (selectedCount === 1) {
            const selectedOptText = selectedOptions[0].text;
            config.button.textContent = selectedOptText.length > 20 ? selectedOptText.substring(0,18) + "..." : selectedOptText;
        } else {
            config.button.textContent = `${selectedCount} selected`;
        }
    }

    function getSelectedValuesFromCustomDropdown(config) {
        return config.optionsData.filter(opt => opt.checked).map(opt => opt.value);
    }

    function toggleDropdown(config, forceClose = false) {
        const isOpen = config.panel.classList.contains('open');
        if (forceClose || isOpen) {
            config.panel.classList.remove('open');
            config.button.classList.remove('open');
        } else {
            Object.values(customFilterConfigs).forEach(otherConfig => {
                if (otherConfig !== config) toggleDropdown(otherConfig, true);
            });
            config.panel.classList.add('open');
            config.button.classList.add('open');
        }
    }

    // Additional event listeners for new heatmap controls
    function setupAdditionalHeatmapListeners() {
        // Auto-optimize toggle
        const autoOptimizeBtn = document.getElementById('heatmap-auto-optimize');
        if (autoOptimizeBtn) {
            autoOptimizeBtn.addEventListener('click', () => {
                heatmapConfig.autoOptimize = !heatmapConfig.autoOptimize;
                autoOptimizeBtn.textContent = heatmapConfig.autoOptimize ? 'Auto-Optimize On' : 'Auto-Optimize Off';
                autoOptimizeBtn.classList.toggle('active', heatmapConfig.autoOptimize);
                
                if (heatmapConfig.autoOptimize && currentHeatmapType !== 'none' && lastHeatmapData) {
                    renderCurrentHeatmap();
                }
            });
        }

        // Smooth transitions toggle
        const smoothTransitionsBtn = document.getElementById('heatmap-smooth-transitions');
        if (smoothTransitionsBtn) {
            smoothTransitionsBtn.addEventListener('click', () => {
                heatmapConfig.smoothTransitions = !heatmapConfig.smoothTransitions;
                smoothTransitionsBtn.textContent = heatmapConfig.smoothTransitions ? 'Smooth On' : 'Smooth Off';
                smoothTransitionsBtn.classList.toggle('active', heatmapConfig.smoothTransitions);
            });
        }

        // Reset parameters button
        const resetParamsBtn = document.getElementById('heatmap-reset-params');
        if (resetParamsBtn) {
            resetParamsBtn.addEventListener('click', () => {
                // Reset to auto-optimize mode
                heatmapConfig.autoOptimize = true;
                heatmapConfig.smoothTransitions = true;
                
                // Update button states
                if (autoOptimizeBtn) {
                    autoOptimizeBtn.textContent = 'Auto-Optimize On';
                    autoOptimizeBtn.classList.add('active');
                }
                
                if (smoothTransitionsBtn) {
                    smoothTransitionsBtn.textContent = 'Smooth On';
                    smoothTransitionsBtn.classList.add('active');
                }
                
                // Reset base configuration
                heatmapConfig.baseRadius = 25;
                heatmapConfig.baseBlur = 15;
                heatmapConfig.maxIntensity = 1.0;
                
                // Re-render heatmap with optimal parameters
                if (currentHeatmapType !== 'none' && lastHeatmapData) {
                    renderCurrentHeatmap();
                }
            });
        }
    }

    function setupEventListeners() {
        applyFiltersBtn.addEventListener('click', fetchAndDisplayMapData);
        cityEl.addEventListener('change', (e) => {
            updateCityDependentCustomFilters(e.target.value);
        });
        areaMainTypeEl.addEventListener('change', () => {
            updateAreaSubTypeCustomFilter();
            // Note: Grid visualization controls are now always visible
            // They will be functional when Coverage Grid is selected
        });
        
        // Lat/Lng finder
        btnFindLocation.addEventListener('click', () => {
            const lat = parseFloat(latFinderInputEl.value);
            const lng = parseFloat(lngFinderInputEl.value);
            if (isNaN(lat) || isNaN(lng)) {
                alert('Please enter valid numbers for Latitude and Longitude.');
                return;
            }
            if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
                alert('Latitude must be between -90 and 90.\nLongitude must be between -180 and 180.');
                return;
            }
            // Remove previous temporary marker if it exists
            if (tempLocationMarker) {
                map.removeLayer(tempLocationMarker);
            }
            // Fly to the new location with a closer zoom
            map.flyTo([lat, lng], 16);
            // Add a distinct marker for the found location
            tempLocationMarker = L.circleMarker([lat, lng], {
                color: '#f44336', // Use a distinct red color
                radius: 10,
                weight: 3,
                opacity: 1,
                fillOpacity: 0.5,
                pane: 'markerPane' // Ensure it's on top
            }).bindPopup(`<b>Location</b><br>Lat: ${lat}<br>Lng: ${lng}`).addTo(map).openPopup();
        });
        
        // Updated map type buttons
        Object.keys(mapTypeButtons).forEach(type => {
            const button = mapTypeButtons[type];
            button.addEventListener('click', () => {
                if (type === 'vendors') {
                    // Simple vendors map - no reset, just switch to vendor view
                    console.log("Vendors Map button clicked - switching to vendor view...");
                    currentHeatmapType = 'none';
                    setActiveMapTypeButton(type);
                    fetchAndDisplayMapData(); // Simple fetch with current filters
                } else {
                    const typeMapping = {
                        densityTotal: 'order_density',
                        densityOrganic: 'order_density_organic',
                        densityNonOrganic: 'order_density_non_organic',
                        userDensity: 'user_density',
                        population: 'population'
                    };
                    currentHeatmapType = typeMapping[type] || 'none';
                    setActiveMapTypeButton(type);
                    fetchAndDisplayMapData();
                }
            });
        });
        
        btnToggleVendors.addEventListener('click', () => {
            vendorsAreVisible = !vendorsAreVisible;
            btnToggleVendors.textContent = vendorsAreVisible ? 'Vendors On' : 'Vendors Off';
            btnToggleVendors.classList.toggle('active', vendorsAreVisible);
            if (vendorsAreVisible) {
                if (!map.hasLayer(vendorLayerGroup)) map.addLayer(vendorLayerGroup);
            } else {
                if (map.hasLayer(vendorLayerGroup)) map.removeLayer(vendorLayerGroup);
            }
        });
        
        btnClearHeatmap.addEventListener('click', () => {
            currentHeatmapType = 'none';
            setActiveMapTypeButton(null);
            if (heatmapLayer) { map.removeLayer(heatmapLayer); heatmapLayer = null; }
        });
        
        Object.values(customFilterConfigs).forEach(config => {
            config.button.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleDropdown(config);
            });
        });
        
        document.addEventListener('click', (e) => {
            Object.values(customFilterConfigs).forEach(config => {
                if (config.button && !config.button.contains(e.target) && config.panel && !config.panel.contains(e.target)) {
                    toggleDropdown(config, true);
                }
            });
        });
        
        vendorRadiusToggleBtn.addEventListener('click', () => {
            showVendorRadius = !showVendorRadius;
            vendorRadiusToggleBtn.textContent = showVendorRadius ? 'Hide Radius' : 'Show Radius';
            vendorRadiusToggleBtn.classList.toggle('active', showVendorRadius);
            redrawVendorRadii();
        });
        
        // Radius modifier controls
        radiusModeSelector.addEventListener('change', (e) => {
            currentRadiusMode = e.target.value;
            if (currentRadiusMode === 'fixed') {
                radiusPercentageControl.style.display = 'none';
                radiusFixedControl.style.display = 'block';
            } else {
                radiusPercentageControl.style.display = 'block';
                radiusFixedControl.style.display = 'none';
            }
        });
        
        vendorRadiusModifierEl.addEventListener('input', (e) => {
            const value = e.target.value;
            radiusModifierValueEl.textContent = value;
            currentRadiusModifier = parseInt(value) / 100;
        });
        
        vendorRadiusFixedEl.addEventListener('input', (e) => {
            const value = e.target.value;
            radiusFixedValueEl.textContent = value;
            currentRadiusFixed = parseFloat(value);
        });
        
        btnResetRadius.addEventListener('click', () => {
            vendorRadiusModifierEl.value = "100";
            radiusModifierValueEl.textContent = "100";
            currentRadiusModifier = 1.0;
            vendorRadiusFixedEl.value = "3";
            radiusFixedValueEl.textContent = "3";
            currentRadiusFixed = 3.0;
            fetchAndDisplayMapData(); // Re-fetch to apply original radius
        });
        
        // Grid visualization controls
        gridBlurEl.addEventListener('input', (e) => {
            gridBlurValueEl.textContent = e.target.value;
            applyGridVisualizationEffects();
        });
        
        gridFadeEl.addEventListener('input', (e) => {
            gridFadeValueEl.textContent = e.target.value;
            applyGridVisualizationEffects();
        });
        
        gridPointSizeEl.addEventListener('input', (e) => {
            gridPointSizeValueEl.textContent = e.target.value;
            // Re-draw grid with new point size
            if (areaMainTypeEl.value === 'coverage_grid') {
                drawCoverageGrid();
            }
        });
        
        // --- MODIFIED Event Listener for marketingAreasOnTopBtn ---
        marketingAreasOnTopBtn.addEventListener('click', () => {
            marketingAreasOnTop = !marketingAreasOnTop;
            marketingAreasOnTopBtn.textContent = marketingAreasOnTop ? 'On' : 'Off';
            marketingAreasOnTopBtn.classList.toggle('active', marketingAreasOnTop);
    
            // This button now directly controls the visibility of the polygon layer.
            if (marketingAreasOnTop) {
                // If toggled ON, add the polygon layer to the map.
                // restylePolygons will ensure it's drawn with the latest data and styles.
                restylePolygons();
                if (!map.hasLayer(polygonLayerGroup)) {
                    map.addLayer(polygonLayerGroup);
                }
            } else {
                // If toggled OFF, remove the polygon layer from the map.
                if (map.hasLayer(polygonLayerGroup)) {
                    map.removeLayer(polygonLayerGroup);
                }
            }
            // Always update the z-index to ensure correct layering when visible.
            updateLayerOrder();
        });
        
        radiusEdgeColorEl.addEventListener('input', redrawVendorRadii);
        radiusInnerColorEl.addEventListener('input', redrawVendorRadii);
        radiusInnerNoneEl.addEventListener('change', redrawVendorRadii);
        areaFillColorEl.addEventListener('input', restylePolygons);
        areaFillNoneEl.addEventListener('change', restylePolygons);
        vendorMarkerSizeEl.addEventListener('input', (e) => updateVendorIconSize(e.target.value));
        
        // Enhanced heatmap control event listeners
        heatmapRadiusEl.addEventListener('input', (e) => {
            heatmapRadiusValueEl.textContent = e.target.value;
            heatmapConfig.autoOptimize = false; // Disable auto-optimize when user adjusts manually
            heatmapConfig.baseRadius = parseInt(e.target.value); // Update base value
            
            // Update auto-optimize button state
            const autoOptimizeBtn = document.getElementById('heatmap-auto-optimize');
            if (autoOptimizeBtn) {
                autoOptimizeBtn.textContent = 'Auto-Optimize Off';
                autoOptimizeBtn.classList.remove('active');
            }
            
            renderCurrentHeatmap();
        });

        heatmapBlurEl.addEventListener('input', (e) => {
            heatmapBlurValueEl.textContent = e.target.value;
            heatmapConfig.autoOptimize = false;
            heatmapConfig.baseBlur = parseInt(e.target.value);
            
            const autoOptimizeBtn = document.getElementById('heatmap-auto-optimize');
            if (autoOptimizeBtn) {
                autoOptimizeBtn.textContent = 'Auto-Optimize Off';
                autoOptimizeBtn.classList.remove('active');
            }
            
            renderCurrentHeatmap();
        });

        heatmapMaxValEl.addEventListener('input', (e) => {
            heatmapMaxValValueEl.textContent = e.target.value;
            heatmapConfig.autoOptimize = false;
            heatmapConfig.maxIntensity = parseFloat(e.target.value);
            
            const autoOptimizeBtn = document.getElementById('heatmap-auto-optimize');
            if (autoOptimizeBtn) {
                autoOptimizeBtn.textContent = 'Auto-Optimize Off';
                autoOptimizeBtn.classList.remove('active');
            }
            
            renderCurrentHeatmap();
        });

        // Setup additional controls
        setupAdditionalHeatmapListeners();
    }
    
    function setActiveMapTypeButton(activeTypeKey) {
        Object.values(mapTypeButtons).forEach(btn => btn.classList.remove('active-map-type'));
        if (activeTypeKey && mapTypeButtons[activeTypeKey]) {
            mapTypeButtons[activeTypeKey].classList.add('active-map-type');
        }
    }

    function showLoading(isLoading, message = 'LOADING ...') {
        if (isLoading) {
            bodyEl.classList.add('is-loading');
            globalLoadingOverlayEl.textContent = message;
            globalLoadingOverlayEl.classList.add('visible');
        } else {
            bodyEl.classList.remove('is-loading');
            globalLoadingOverlayEl.classList.remove('visible');
        }
    }
    
    // NEW: Enhanced fetchAndDisplayMapData function (add zoom level parameter)
    async function fetchAndDisplayMapData() {
        // Clear the temporary marker on new search
        if (tempLocationMarker) {
            map.removeLayer(tempLocationMarker);
            tempLocationMarker = null;
        }

        const params = new URLSearchParams();
        const isCoverageGrid = areaMainTypeEl.value === 'coverage_grid';
        const selectedCity = cityEl.value;
        const selectedBLs = getSelectedValuesFromCustomDropdown(customFilterConfigs.businessLine);

        if (isCoverageGrid && selectedCity === 'tehran') {
            if (selectedBLs.length !== 1) {
                showLoading(false);
                alert('Target-based Coverage Analysis requires selecting exactly ONE Business Line.');
                if(applyFiltersBtn) applyFiltersBtn.disabled = false;
                return;
            }
        }

        params.append('city', cityEl.value);
        params.append('start_date', daterangeStartEl.value);
        params.append('end_date', daterangeEndEl.value);
        params.append('area_type_display', areaMainTypeEl.value);
        
        // NEW: Add current zoom level to params for backend optimization
        params.append('zoom_level', map.getZoom().toString());

        getSelectedValuesFromCustomDropdown(customFilterConfigs.areaSubType)
            .forEach(val => params.append(customFilterConfigs.areaSubType.paramName, val));
        getSelectedValuesFromCustomDropdown(customFilterConfigs.businessLine)
            .forEach(val => params.append(customFilterConfigs.businessLine.paramName, val));

        const vendorCodesText = vendorCodesFilterEl.value.trim();
        if (vendorCodesText) params.append('vendor_codes_filter', vendorCodesText);

        getSelectedValuesFromCustomDropdown(customFilterConfigs.vendorStatus)
            .forEach(val => params.append(customFilterConfigs.vendorStatus.paramName, val));
        getSelectedValuesFromCustomDropdown(customFilterConfigs.vendorGrade)
            .forEach(val => params.append(customFilterConfigs.vendorGrade.paramName, val));

        params.append('vendor_visible', vendorVisibleEl.value);
        params.append('vendor_is_open', vendorIsOpenEl.value);
        params.append('heatmap_type_request', currentHeatmapType);
        params.append('radius_modifier', currentRadiusModifier);
        params.append('radius_mode', currentRadiusMode);
        params.append('radius_fixed', currentRadiusFixed);

        console.log("Fetching map data with params:", params.toString());
        showLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/map-data?${params.toString()}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: "Unknown server error", details: response.statusText}));
                showLoading(true, `Error: ${errorData.error || response.statusText}. Check console.`);
                throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
            }

            const data = await response.json();
            if(data.error){
                showLoading(true, `Backend error: ${data.error}. Check console.`);
                throw new Error(`Backend error: ${data.error} - ${data.details || ''}`);
            }

            allVendorsData = data.vendors || [];
            allPolygonsData = data.polygons || null;
            lastHeatmapData = data.heatmap_data || null;
            allCoverageGridData = data.coverage_grid || [];

            // Debug logging for heatmap data
            if (lastHeatmapData && lastHeatmapData.length > 0) {
                console.log(`Received ${lastHeatmapData.length} heatmap points for type: ${currentHeatmapType}`);
                const values = lastHeatmapData.map(p => p.value).filter(v => v != null);
                if (values.length > 0) {
                    console.log(`Value range: ${Math.min(...values).toFixed(2)} - ${Math.max(...values).toFixed(2)}`);
                }
            }

            updateMapLayers();
            showLoading(false);
        } catch (error) {
            console.error("Fetch/Display Error:", error);
            if (!bodyEl.classList.contains('is-loading') || globalLoadingOverlayEl.textContent === 'LOADING ...') {
                showLoading(true, `Error: ${error.message}. Check console or try refreshing.`);
            }
        }
    }

    // --- MODIFIED updateMapLayers function ---
    function updateMapLayers() {
        console.log(`[DEBUG] updateMapLayers called. vendorsAreVisible: ${vendorsAreVisible}`);
        
        redrawVendorMarkersAndRadii();
        
        // Ensure vendor layer is added to map if vendors are visible
        console.log(`[DEBUG] Before layer management - vendorLayerGroup on map: ${map.hasLayer(vendorLayerGroup)}`);
        
        if (vendorsAreVisible && !map.hasLayer(vendorLayerGroup)) {
            console.log(`[DEBUG] Adding vendor layer to map`);
            map.addLayer(vendorLayerGroup);
        } else if (!vendorsAreVisible && map.hasLayer(vendorLayerGroup)) {
            console.log(`[DEBUG] Removing vendor layer from map`);
            map.removeLayer(vendorLayerGroup);
        } else {
            console.log(`[DEBUG] No layer change needed. vendorsAreVisible: ${vendorsAreVisible}, hasLayer: ${map.hasLayer(vendorLayerGroup)}`);
        }
        
        console.log(`[DEBUG] After layer management - vendorLayerGroup on map: ${map.hasLayer(vendorLayerGroup)}`);
        
        restylePolygons(); // This always prepares the polygonLayerGroup with data and styles
        drawCoverageGrid();
        renderCurrentHeatmap();
        // --- NEW VISIBILITY LOGIC ---
        // When displaying the grid, polygons should be hidden by default.
        // The 'On Top' button is responsible for showing them.
        if (areaMainTypeEl.value === 'coverage_grid') {
            // If the toggle is OFF, ensure the layer is removed from the map.
            if (!marketingAreasOnTop && map.hasLayer(polygonLayerGroup)) {
                map.removeLayer(polygonLayerGroup);
            }
        } else {
            // For any other display type, ensure polygons are added back to the map.
            if (!map.hasLayer(polygonLayerGroup)) {
                map.addLayer(polygonLayerGroup);
            }
        }
        adjustMapView();
        applyGridVisualizationEffects();
        updateLayerOrder();
    }
    
    // NEW: Enhanced renderCurrentHeatmap function
    function renderCurrentHeatmap() {
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }

        if (currentHeatmapType === 'none' || !lastHeatmapData || lastHeatmapData.length === 0) {
            return;
        }

        // Filter out invalid data points
        const validData = lastHeatmapData.filter(p => 
            p.lat != null && p.lng != null && p.value != null && 
            !isNaN(p.lat) && !isNaN(p.lng) && !isNaN(p.value) &&
            p.value > 0
        );

        if (validData.length === 0) {
            console.warn('No valid heatmap data after filtering');
            return;
        }

        currentZoomLevel = map.getZoom();
        
        // Get optimized parameters
        const optimalParams = heatmapConfig.autoOptimize ? 
            calculateOptimalHeatmapParams(validData, currentZoomLevel) : 
            getZoomAdjustedHeatmapOptions();

        lastOptimalParams = optimalParams;

        // Update UI to show optimal parameters
        if (heatmapConfig.autoOptimize) {
            updateHeatmapControlsDisplay(optimalParams);
        }

        // Prepare heatmap data with proper intensity scaling
        const heatPoints = validData.map(p => {
            // Ensure values are properly scaled for the heatmap
            const intensity = Math.max(0.1, Math.min(100, p.value)) / 100;
            return [p.lat, p.lng, intensity];
        });

        // Enhanced heatmap options
        let heatOptions = {
            radius: optimalParams.radius,
            blur: optimalParams.blur,
            max: optimalParams.max,
            minOpacity: 0.3,
            maxZoom: 18, // Keep this but rely more on our zoom-adjusted scaling
            pane: 'overlayPane'  // Ensure proper layering
        };

        // Enhanced gradients for different heatmap types
        if (currentHeatmapType === 'order_density') {
            heatOptions.gradient = {
                0.0: 'rgba(0, 0, 255, 0)',
                0.15: 'rgba(0, 100, 255, 0.4)',
                0.3: 'rgba(0, 200, 255, 0.6)',
                0.5: 'rgba(0, 255, 100, 0.8)',
                0.7: 'rgba(255, 255, 0, 0.9)',
                1.0: 'rgba(255, 0, 0, 1)'
            };
        } else if (currentHeatmapType === 'order_density_organic') {
            heatOptions.gradient = {
                0.0: 'rgba(0, 128, 0, 0)',
                0.2: 'rgba(0, 200, 0, 0.5)',
                0.4: 'rgba(100, 255, 0, 0.7)',
                0.6: 'rgba(200, 255, 0, 0.8)',
                0.8: 'rgba(255, 200, 0, 0.9)',
                1.0: 'rgba(255, 100, 0, 1)'
            };
        } else if (currentHeatmapType === 'order_density_non_organic') {
            heatOptions.gradient = {
                0.0: 'rgba(128, 0, 128, 0)',
                0.2: 'rgba(150, 0, 150, 0.5)',
                0.4: 'rgba(200, 0, 200, 0.7)',
                0.6: 'rgba(255, 0, 150, 0.8)',
                0.8: 'rgba(255, 50, 100, 0.9)',
                1.0: 'rgba(255, 0, 0, 1)'
            };
        } else if (currentHeatmapType === 'user_density') {
            heatOptions.gradient = {
                0.0: 'rgba(75, 0, 130, 0)',
                0.2: 'rgba(100, 50, 200, 0.5)',
                0.4: 'rgba(150, 100, 255, 0.7)',
                0.6: 'rgba(200, 150, 255, 0.8)',
                0.8: 'rgba(255, 200, 150, 0.9)',
                1.0: 'rgba(255, 100, 0, 1)'
            };
        } else {
            // Default gradient for population and others
            heatOptions.gradient = {
                0.0: 'rgba(0, 0, 255, 0)',
                0.25: 'rgba(0, 255, 255, 0.6)',
                0.5: 'rgba(0, 255, 0, 0.8)',
                0.75: 'rgba(255, 255, 0, 0.9)',
                1.0: 'rgba(255, 0, 0, 1)'
            };
        }

        console.log(`Rendering heatmap: ${validData.length} points, radius: ${optimalParams.radius}, blur: ${optimalParams.blur}, max: ${optimalParams.max.toFixed(2)}`);

        heatmapLayer = L.heatLayer(heatPoints, heatOptions).addTo(map);

        // Make heatmap non-interactive
        if (heatmapLayer && heatmapLayer.getPane()) {
            heatmapLayer.getPane().style.pointerEvents = 'none';
        }
    }
    
    function applyGridVisualizationEffects() {
        const blur = gridBlurEl.value;
        const fade = gridFadeEl.value / 100;
        
        // Apply CSS filter to the coverage grid pane
        const coveragePane = map.getPane('coverageGridPane');
        if (coveragePane) {
            coveragePane.style.filter = `blur(${blur}px)`;
            coveragePane.style.opacity = fade;
        }
    }
    
    function updateLayerOrder() {
        if (marketingAreasOnTop) {
            // Move marketing areas (polygons) on top of grid
            map.getPane('polygonPane').style.zIndex = 470;
            map.getPane('coverageGridPane').style.zIndex = 460;
        } else {
            // Default order - grid on top of polygons
            map.getPane('polygonPane').style.zIndex = 450;
            map.getPane('coverageGridPane').style.zIndex = 460;
        }
    }
    
    function drawCoverageGrid() {
        coverageGridLayerGroup.clearLayers();
        if (areaMainTypeEl.value !== 'coverage_grid' || !allCoverageGridData || allCoverageGridData.length === 0) {
            return;
        }
        
        // Get current grid point size
        const gridPointSize = parseInt(gridPointSizeEl.value) || 6;
        
        allCoverageGridData.forEach(point => {
            let color = '#808080'; // Default to neutral grey
            let popupContent = `<b>Coverage at (${point.lat.toFixed(4)}, ${point.lng.toFixed(4)})</b><br>`;
            if (point.marketing_area) {
                popupContent += `<b>Marketing Area:</b> ${decodeURIComponentSafe(point.marketing_area)}<br>`;
            }
            if (point.target_business_line && point.target_value != null) {
                const { actual_value, target_value, performance_ratio } = point;
                // Coloring logic based on performance ratio (actual / target)
                if (actual_value === 0) {
                    color = '#d3d3d3'; // Light Grey (Zero Coverage)
                } else if (performance_ratio >= 1.2) {
                    color = '#004d00'; // Darker Green (Exceeds Target by 20%+)
                } else if (performance_ratio >= 1.0) {
                    color = '#00FF00'; // Bright Green (Meets or slightly exceeds Target)
                } else if (performance_ratio >= 0.8) {
                    color = '#ffff00'; // Yellow (Good, 80%-99%)
                } else if (performance_ratio >= 0.6) {
                    color = '#ff9900'; // Orange (Okay, 60%-79%)
                } else if (performance_ratio >= 0.4) {
                    color = '#ff0000'; // Red (Poor, 40%-59%)
                } else if (performance_ratio >= 0.2) {
                    color = '#8B0000'; // Dark Red (Very Poor, 20%-39%)
                } else {
                    color = '#000000'; // Black (Extremely Poor, <20%)
                }
                // Build a detailed popup for target-based analysis
                popupContent += `<hr style="margin: 4px 0;"><b>Target Analysis (${point.target_business_line})</b><br>`;
                popupContent += `<b>Target Count:</b> ${target_value}<br>`;
                popupContent += `<b>Actual Count:</b> ${actual_value}<br>`;
                popupContent += `<b>Performance:</b> <b style="color:${color};">${(performance_ratio * 100).toFixed(0)}%</b><br>`;
            
            } else {
                // This block is now for points that exist but have no applicable target data
                // for the selected business line in their area.
                popupContent += `<br><i>No target data found for this Business Line in this area.</i>`;
            }
            // Add general coverage details to all popups
            const coverage = point.coverage;
            popupContent += `<hr style="margin: 4px 0;"><b>Total Covering Vendors:</b> ${coverage.total_vendors}<br>`;
            if (coverage.by_business_line && Object.keys(coverage.by_business_line).length > 0) {
                popupContent += '<b>By Business Line:</b><br>';
                Object.entries(coverage.by_business_line)
                    .sort((a, b) => b[1] - a[1]) // Sort by count descending
                    .forEach(([bl, count]) => {
                        popupContent += `  ${bl}: ${count}<br>`;
                    });
            }
            if (coverage.by_grade && Object.keys(coverage.by_grade).length > 0) {
                popupContent += '<b>By Grade:</b><br>';
                Object.entries(coverage.by_grade)
                    .sort((a,b) => b[1] - a[1])
                    .forEach(([grade, count]) => {
                        popupContent += `  ${grade}: ${count}<br>`;
                    });
            }
            const marker = L.circleMarker([point.lat, point.lng], {
                radius: gridPointSize,
                fillColor: color,
                color: '#333',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.8,
                pane: 'coverageGridPane'
            });
            marker.bindPopup(popupContent);
            coverageGridLayerGroup.addLayer(marker);
        });
    }

    function adjustMapView() {
        let bounds;
        const hasVisibleVendors = vendorsAreVisible && vendorLayerGroup.getLayers().length > 0;
        const hasVisiblePolygons = map.hasLayer(polygonLayerGroup) && polygonLayerGroup.getLayers().length > 0;
        const hasVisibleCoverage = coverageGridLayerGroup.getLayers().length > 0;
        
        let allVisibleLayers = L.featureGroup();
        if (hasVisibleVendors) allVisibleLayers.addLayer(vendorLayerGroup);
        if (hasVisiblePolygons) allVisibleLayers.addLayer(polygonLayerGroup);
        if (hasVisibleCoverage) allVisibleLayers.addLayer(coverageGridLayerGroup);
        if (allVisibleLayers.getLayers().length > 0) {
            bounds = allVisibleLayers.getBounds();
        }
        
        if (bounds && bounds.isValid()) {
            map.fitBounds(bounds, {padding: [50, 50]});
        } else if (cityEl.value === "tehran") {
            map.setView([35.7219, 51.3347], 11);
        } else if (cityEl.value === "mashhad") {
            map.setView([36.297, 59.606], 12);
        } else if (cityEl.value === "shiraz") {
            map.setView([29.5918, 52.5837], 12);
        }
    }

    function redrawVendorMarkersAndRadii() {
        console.log(`[DEBUG] redrawVendorMarkersAndRadii called. vendorsAreVisible: ${vendorsAreVisible}, allVendorsData: ${allVendorsData ? allVendorsData.length : 0} vendors`);
        
        vendorLayerGroup.clearLayers(); 
        if (!allVendorsData || allVendorsData.length === 0) {
            console.log(`[DEBUG] No vendor data available`);
            return;
        }
        
        let markersCreated = 0;
        allVendorsData.forEach(vendor => {
            if (vendor.latitude == null || vendor.longitude == null) return; 
            const popupContent = `<b>${vendor.vendor_name || 'N/A'}</b><br>
                                Code: ${vendor.vendor_code || 'N/A'}<br>
                                Status: ${vendor.status_id !== null ? vendor.status_id : 'N/A'}<br>
                                Grade: ${vendor.grade || 'N/A'}<br>
                                Visible: ${vendor.visible == 1 ? 'Yes' : (vendor.visible == 0 ? 'No' : 'N/A')}<br>
                                Open: ${vendor.open == 1 ? 'Yes' : (vendor.open == 0 ? 'No' : 'N/A')}<br>
                                Radius: ${vendor.radius ? vendor.radius.toFixed(2) + ' km' : 'N/A'}`;
            const marker = L.marker([vendor.latitude, vendor.longitude], {icon: defaultVendorIcon})
                .bindPopup(popupContent);
            vendorLayerGroup.addLayer(marker);
            markersCreated++;
        });
        
        console.log(`[DEBUG] Created ${markersCreated} vendor markers`);
        console.log(`[DEBUG] Vendor layer group has ${vendorLayerGroup.getLayers().length} layers`);
        console.log(`[DEBUG] Vendor layer is on map: ${map.hasLayer(vendorLayerGroup)}`);
        
        redrawVendorRadii(); 
    }
    
    function redrawVendorRadii() {
        // This is inefficient. A better approach is to keep references.
        const circles = [];
        vendorLayerGroup.eachLayer(layer => {
            if (layer instanceof L.Circle) {
                circles.push(layer);
            }
        });
        circles.forEach(circle => vendorLayerGroup.removeLayer(circle));
        if (!showVendorRadius || !allVendorsData) return; 
        const rEdgeColor = radiusEdgeColorEl.value;
        const rInnerIsNone = radiusInnerNoneEl.checked;
        const rInnerColor = rInnerIsNone ? 'transparent' : radiusInnerColorEl.value;
        allVendorsData.forEach(vendor => {
            if (vendor.latitude != null && vendor.longitude != null && vendor.radius > 0) {
                 L.circle([vendor.latitude, vendor.longitude], {
                    radius: vendor.radius * 1000, // radius is already modified by backend
                    color: rEdgeColor, 
                    fillColor: rInnerColor, 
                    fillOpacity: rInnerIsNone ? 0 : 0.25, 
                    weight: 1.5,
                    pane: 'shadowPane' // Render circles behind markers AND polygons
                }).addTo(vendorLayerGroup); 
            }
        });
    }

    function restylePolygons() {
        polygonLayerGroup.clearLayers();
        if (!allPolygonsData || !allPolygonsData.features || allPolygonsData.features.length === 0) return;
        const polyFillIsNone = areaFillNoneEl.checked;
        const polyFillColor = polyFillIsNone ? 'transparent' : areaFillColorEl.value;
        
        L.geoJSON(allPolygonsData, {
            pane: 'polygonPane',
            style: (feature) => {
                let defaultStyle = {
                    color: "#1E88E5", weight: 1.5, opacity: 0.7,
                    fillColor: polyFillColor, fillOpacity: polyFillIsNone ? 0 : 0.3
                };
                // When we're displaying polygons with no enrichment (like the overlay), just use the default.
                if (!feature.properties || Object.keys(feature.properties).length <= 2) { // e.g. only name/id and geometry
                    return defaultStyle;
                }
                return defaultStyle; // For now, we don't have a special style based on properties
            },
            onEachFeature: (feature, layer) => {
                let popupContent = '';
                if (feature.properties) {
                    const p = feature.properties;
                    const nameCand = p.name || p.NAME || p.Name || p.Region || p.REGION_N || p.NAME_1 || p.NAME_2 || p.district || p.NAME_MAHAL;
                    const name = decodeURIComponentSafe(nameCand) || "Area Detail";
                    popupContent += `<b>${name}</b>`;
                    // Population Data Section (for Tehran districts)
                    if (p.Pop != null || p.PopDensity != null) {
                        popupContent += `<br><hr style="margin: 5px 0; border-color: #eee;"><em>Population Stats:</em>`;
                        if (p.Pop != null) {
                            popupContent += `<br><b>Population:</b> ${Number(p.Pop).toLocaleString()}`;
                        }
                        if (p.PopDensity != null) {
                             popupContent += `<br><b>Population Density:</b> ${Number(p.PopDensity).toFixed(2)}/km²`;
                        }
                    }
                    
                    // Filter-Based Data Section (applies to all polygon types)
                    if (p.vendor_count != null || p.unique_user_count != null) {
                        popupContent += `<br><hr style="margin: 5px 0; border-color: #eee;"><em>Metrics (based on filters):</em>`;
                        
                        // Vendor Metrics
                        if (p.vendor_count != null) {
                             popupContent += `<br><b>Total Filtered Vendors:</b> ${p.vendor_count}`;
                        }
                        if (p.grade_counts && Object.keys(p.grade_counts).length > 0) {
                            popupContent += `<br><b>- By Grade:</b> `;
                            const gradeStrings = Object.entries(p.grade_counts)
                                                      .sort((a, b) => b[1] - a[1]) // Sort by count desc
                                                      .map(([grade, count]) => `${grade}: ${count}`);
                            popupContent += gradeStrings.join(', ');
                        }
                        if (p.vendor_per_10k_pop != null) {
                            popupContent += `<br><b>- Vendors per 10k Pop:</b> ${Number(p.vendor_per_10k_pop).toFixed(2)}`;
                        }
                        // Customer Metrics
                        if (p.unique_user_count != null) {
                             popupContent += `<br><b>Unique Customers (date range):</b> ${p.unique_user_count.toLocaleString()}`;
                        }
                        if (p.total_unique_user_count != null) {
                             popupContent += `<br><b>Unique Customers (all time):</b> ${p.total_unique_user_count.toLocaleString()}`;
                        }
                    } else if (Object.keys(p).length <= 4) { // Heuristic to check if it's an un-enriched polygon
                         // Don't show the "Metrics" header if there are no metrics to display.
                    }
                } else {
                    popupContent = '<b>Area</b>';
                }
                layer.bindPopup(popupContent);
            }
        }).addTo(polygonLayerGroup);
    }

    init();
});