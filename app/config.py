TEXT = {
    "EN": {
        "nav_title": "Navigation",
        "page_db": "Database Analysis",
        "page_sim": "Manual Simulator",
        "loading_data": "Loading data...",
        "loading_model": "Loading model...",
        "title_app": "🩸 Blood Donors Prediction System",
        "title_db": "Donor Database Analysis",
        "title_sim": "Donor Simulator",
        "subtitle_sim": "Simulate a donor's path based on manual inputs.",
        "input_gender": "Gender",
        "input_birth": "Birth Year",
        "input_history": "Donation History (Edit values below)",
        "btn_simulate": "Run Simulation",
        "view_data": "View Raw Data & Sociodemographics",
        "chart_title": "Donation History & Forecast",
        "axis_x": "Year",
        "axis_y": "Donations",
        "legend_trend": "Trend",
        "legend_pred": "Prediction (Next Year)",
        "metric_state": "Last State",
        "metric_prob": "Future Prob.",
        "metric_exp": "Expected Value (Lambda)",
        "static_info": "Sociodemographic Info",
        "error_len": "History length mismatch.",
        "state_prefix": "State",
        "tooltip_year": "Year",
        "tooltip_don": "Donations",
        "tooltip_state": "State",
        "search_label": "Search Donor",
        "go_to_manual_simulator": "Go to the manual simulato",
        "state_names": {
            0: "Non-Donor",
            1: "Occasional Donor",
            2: "Frequent Donor",
        },
        "app_description": """
    ### Welcome!
    Please select a module from the navigation menu:

    * **1. Database Analysis:** Analyze historical data and predictions for existing donors.
    * **2. Simulator:** Manually input sociodemographic data and donation history to forecast future behavior.
    """
    },
    "IT": {
        "nav_title": "Navigazione",
        "page_db": "Analisi Database",
        "page_sim": "Simulatore Manuale",
        "loading_data": "Caricamento dati...",
        "loading_model": "Caricamento modello...",
        "title_app": "🩸 Sistema di Previsione Donatori",
        "title_db": "Analisi Database Donatori",
        "title_sim": "Simulatore Donatore",
        "subtitle_sim": "Simula il percorso di un donatore inserendo i dati manualmente.",
        "input_gender": "Genere",
        "input_birth": "Anno di Nascita",
        "input_history": "Storia Donazioni (Modifica i valori sotto)",
        "btn_simulate": "Esegui Simulazione",
        "view_data": "Vedi Dati Grezzi e Sociodemografici",
        "chart_title": "Storia Donazioni e Previsione",
        "axis_x": "Anno",
        "axis_y": "Donazioni",
        "legend_trend": "Andamento",
        "legend_pred": "Predizione (Anno Prossimo)",
        "metric_state": "Ultimo Stato",
        "metric_prob": "Prob. Futura",
        "metric_exp": "Valore Atteso (Lambda)",
        "static_info": "Info Sociodemografiche",
        "error_len": "Lunghezza storico errata.",
        "state_prefix": "Stato",
        "tooltip_year": "Anno",
        "tooltip_don": "Donazioni",
        "tooltip_state": "Stato",
        "search_label": "Cerca Donatore",
        "go_to_manual_simulator": "Vai al simulatore manuale",
        "state_names": {
            0: "Non-Donatore",
            1: "Donatore Saltuario",
            2: "Donatore Frequente",
        },
        "app_description": """
    ### Benvenuto!
    Seleziona un modulo dal menu di navigazione:

    * **1. Analisi Database:** Analizza i dati storici e le previsioni per i donatori esistenti.
    * **2. Simulatore:** Inserisci manualmente dati sociodemografici e storico donazioni per prevedere il comportamento futuro.
    """
    },
}

CONFIG = {
    "DATA_PATH": "data/recent_donations.csv",
    "MODEL_PATH": "models/hmm_glm_full.pt",
    "COVID_YEARS": (2020, 2021, 2022),
    "AGE_BINS": [18, 25, 35, 45, 55, 60, 65, 75],
    "GENDER_MAP": {"M": 0, "F": 1},
}

COLOR_MAP = {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}
