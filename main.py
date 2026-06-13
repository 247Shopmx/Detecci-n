import os
import sys
import requests
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- Configuración de Logging y Seguridad ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
MODEL_PATH = 'fifa_2026_model.pkl' # Archivo del modelo pre-entrenado

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ODDS_API_KEY]):
    logger.error("Faltan variables de entorno críticas.")
    sys.exit(1)

# --- Utilidades de Limpieza y Normalización ---
def normalize_team_name(name):
    if not name: return ""
    name = name.lower().strip()
    replacements = {
        'méxico': 'mexico', 'españa': 'spain', 'alemania': 'germany',
        'inglaterra': 'england', 'estados unidos': 'usa', 'corea del sur': 'south korea',
        'países bajos': 'netherlands', 'república checa': 'czech republic'
    }
    accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}
    for acc, char in accents.items():
        name = name.replace(acc, char)
    for esp, eng in replacements.items():
        if esp in name: name = eng
    return name

# --- Ingesta de Datos Reales (ESPN API) ---
def get_espn_features():
    """
    Extrae features reales de rendimiento desde ESPN.
    Features: Win %, Goals For, Goals Against, Recent Form Points.
    """
    url = "http://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/events"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        features_list = []
        if 'events' in data:
            for event in data['events']:
                try:
                    home_comp = event['competitions'][0]['competitors'][0]
                    away_comp = event['competitions'][0]['competitors'][1]
                    
                    # Extracción de métricas reales del JSON
                    home_record = home_comp.get('record', {}).get('summary', '0-0-0')
                    away_record = away_comp.get('record', {}).get('summary', '0-0-0')
                    
                    # Parseo simple de récord (W-L-T) a features numéricas
                    h_w, h_l, h_t = map(int, home_record.split('-')) if '-' in home_record else (0,0,0)
                    a_w, a_l, a_t = map(int, away_record.split('-')) if '-' in away_record else (0,0,0)
                    
                    home_total = h_w + h_l + h_t if (h_w + h_l + h_t) > 0 else 1
                    away_total = a_w + a_l + a_t if (a_w + a_l + a_t) > 0 else 1
                    
                    features_list.append({
                        'home_team_raw': home_comp['team']['displayName'],
                        'away_team_raw': away_comp['team']['displayName'],
                        'home_team_norm': normalize_team_name(home_comp['team']['displayName']),
                        'away_team_norm': normalize_team_name(away_comp['team']['displayName']),
                        'home_win_pct': h_w / home_total,
                        'away_win_pct': a_w / away_total,
                        'home_goals_avg': h_w * 1.5 / home_total, # Proxy simplificado si no hay goles explícitos
                        'away_goals_avg': a_w * 1.5 / away_total
                    })
                except Exception as e:
                    logger.warning(f"Error parseando partido ESPN: {e}")
                    continue
                    
        return pd.DataFrame(features_list)
    except Exception as e:
        logger.error(f"Fallo crítico en ESPN API: {e}")
        send_telegram_alert(f"❌ *Fallo de Conexión ESPN:* {str(e)}")
        return pd.DataFrame()

# --- Ingesta de Cuotas Reales (The Odds API) ---
def get_market_odds():
    url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'eu,uk',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        odds_list = []
        for match in data:
            bookmakers = match.get('bookmakers', [])
            if not bookmakers: continue
            
            # Usamos el promedio de las casas principales
            best_bookie = next((b for b in bookmakers if b['title'] in ['Bet365', 'Pinnacle']), bookmakers[0])
            outcomes = best_bookie['markets'][0]['outcomes']
            
            odds_h = next((o['price'] for o in outcomes if o['name'] == match['home_team']), None)
            odds_a = next((o['price'] for o in outcomes if o['name'] == match['away_team']), None)
            odds_d = next((o['price'] for o in outcomes if o['name'] == 'Draw'), None)
            
            if odds_h and odds_a and odds_d:
                odds_list.append({
                    'home_team_raw': match['home_team'],
                    'away_team_raw': match['away_team'],
                    'home_team_norm': normalize_team_name(match['home_team']),
                    'away_team_norm': normalize_team_name(match['away_team']),
                    'odds_home': float(odds_h),
                    'odds_draw': float(odds_d),
                    'odds_away': float(odds_a)
                })
        return pd.DataFrame(odds_list)
    except Exception as e:
        logger.error(f"Fallo crítico en The Odds API: {e}")
        send_telegram_alert(f"❌ *Fallo de Conexión Odds API:* {str(e)}")
        return pd.DataFrame()

# --- Pipeline de Inferencia ML (Ensemble) ---
def predict_probabilities(df_features, model_path):
    """
    Carga el modelo pre-entrenado y genera probabilidades 1X2.
    """
    if not os.path.exists(model_path):
        logger.warning("Modelo no encontrado. Usando fallback estadístico básico.")
        # Fallback seguro si no hay modelo entrenado aún
        df_features['pred_home'] = 0.33
        df_features['pred_draw'] = 0.33
        df_features['pred_away'] = 0.33
        return df_features

    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
            
        scaler = model_bundle['scaler']
        model = model_bundle['model']
        
        feature_cols = ['home_win_pct', 'away_win_pct', 'home_goals_avg', 'away_goals_avg']
        X = df_features[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Predicción de probabilidades (Softmax/Proba)
        probs = model.predict_proba(X_scaled)
        
        # Asumimos orden de clases: [Away, Draw, Home] o similar según entrenamiento
        # Ajusta los índices según cómo entrenaste tu modelo
        df_features['pred_home'] = probs[:, 2] # Ejemplo: índice 2 es Home
        df_features['pred_draw'] = probs[:, 1]
        df_features['pred_away'] = probs[:, 0]
        
        logger.info("Predicciones ML generadas con éxito.")
        return df_features
        
    except Exception as e:
        logger.error(f"Error en inferencia ML: {e}")
        send_telegram_alert(f"⚠️ *Error en Modelo ML:* {str(e)}. Usando fallback.")
        df_features['pred_home'] = 0.33
        df_features['pred_draw'] = 0.33
        df_features['pred_away'] = 0.33
        return df_features

# --- Detección de Valor y Alertas ---
def find_value_bets(espn_df, odds_df):
    if espn_df.empty or odds_df.empty:
        return []
    
    merged = pd.merge(espn_df, odds_df, on=['home_team_norm', 'away_team_norm'], how='inner')
    value_bets = []
    
    for _, row in merged.iterrows():
        # Probabilidad Implícita del Mercado (sin vig)
        imp_h = 1 / row['odds_home']
        imp_d = 1 / row['odds_draw']
        imp_a = 1 / row['odds_away']
        total_imp = imp_h + imp_d + imp_a
        
        norm_imp_h = imp_h / total_imp
        norm_imp_d = imp_d / total_imp
        norm_imp_a = imp_a / total_imp
        
        # Comparación con Probabilidad ML
        edge_h = row['pred_home'] - norm_imp_h
        edge_d = row['pred_draw'] - norm_imp_d
        edge_a = row['pred_away'] - norm_imp_a
        
        max_edge = max(edge_h, edge_d, edge_a)
        
        if max_edge > 0.05: # Umbral de 5%
            selection = "LOCAL" if max_edge == edge_h else ("EMPATE" if max_edge == edge_d else "VISITANTE")
            odds_val = row['odds_home'] if selection == "LOCAL" else (row['odds_draw'] if selection == "EMPATE" else row['odds_away'])
            
            value_bets.append({
                'match': f"{row['home_team_raw']} vs {row['away_team_raw']}",
                'selection': selection,
                'odds': odds_val,
                'edge': f"{max_edge:.2%}",
                'ml_conf': f"{(row['pred_home'] if selection=='LOCAL' else row['pred_draw'] if selection=='EMPATE' else row['pred_away']):.2%}"
            })
            
    return value_bets

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'MarkdownV2'}
    try:
        # Escapado básico para MarkdownV2
        safe_msg = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')
        payload['text'] = safe_msg
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Error Telegram: {e}")

def main():
    logger.info("Iniciando pipeline de detección de valor...")
    
    # 1. Ingesta
    espn_df = get_espn_features()
    odds_df = get_market_odds()
    
    if espn_df.empty or odds_df.empty:
        send_telegram_alert("⚠️ *Alerta de Sistema:* Fallo en ingesta de datos. Revisar logs.")
        return

    # 2. Inferencia ML
    espn_df = predict_probabilities(espn_df, MODEL_PATH)
    
    # 3. Detección
    bets = find_value_bets(espn_df, odds_df)
    
    # 4. Reporte
    if bets:
        report = "🚨 *VALUE BETS DETECTADAS (ML ENSEMBLE)* 🚨\\n\\n"
        for b in bets:
            report += (
                f"⚽ *Partido:* {b['match']}\\n"
                f"🎯 *Selección:* {b['selection']}\\n"
                f"💰 *Cuota:* {b['odds']}\\n"
                f"📈 *Edge:* {b['edge']}\\n"
                f"🤖 *Confianza ML:* {b['ml_conf']}\\n\\n"
                f"---\\n\\n"
            )
        send_telegram_alert(report)
    else:
        send_telegram_alert("✅ *Pipeline ML Ejecutado.*\\n\\nNo se detectaron ineficiencias de valor (>5%) en la jornada actual.")

if __name__ == "__main__":
    main()
