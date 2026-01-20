"""
Flask API para servir análises em tempo real
Executa: python app.py
Acessa: http://localhost:5000
"""

from flask import Flask, jsonify, render_template_string,send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import schedule
import threading
from sentiment_analyzer import MarketSentimentAnalyzer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pasta raiz do projeto
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_FILE = os.path.join(FRONTEND_DIR, "market_sentiment_data.json")
UPDATE_INTERVAL_MINUTES = 15
DATA_FILE = 'market_sentiment_data.json'
analyzer = MarketSentimentAnalyzer()
analysis_lock = threading.Lock()

# Armazenar último timestamp de análise
last_update = None

@app.route("/market_sentiment_data.json")
def sentiment_json():
    return send_from_directory(FRONTEND_DIR, "market_sentiment_data.json")

@app.route('/')
def index():
    """Servir dashboard HTML"""
    dashboard_path = os.path.join(FRONTEND_DIR, 'dashboard.html')
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    """Endpoint para obter dados de sentimento"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({'error': 'Nenhum dado disponível. Execute análise primeiro.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def trigger_analysis():
    """Trigger manual para análise"""
    try:
        print("🚀 Análise manual iniciada...")
        report = analyzer.generate_report()
        analyzer.save_report(report, DATA_FILE)
        analyzer.print_summary(report)
        return jsonify({'status': 'success', 'message': 'Análise concluída'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Obter status da última análise"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({
                'status': 'ok',
                'last_update': data.get('timestamp'),
                'data_file': DATA_FILE
            })
        else:
            return jsonify({'status': 'no_data'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def scheduled_analysis():
    """Função para executar análise agendada"""
    print(f"\n⏰ Análise agendada iniciada em {datetime.now()}")
    try:
        with analysis_lock:
            report = analyzer.generate_report()
            analyzer.save_report(report, DATA_FILE)
            print("✅ Análise concluída com sucesso!")
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

def schedule_updates():
    """Agendar atualizações periódicas"""
    schedule.every(UPDATE_INTERVAL_MINUTES).minutes.do(scheduled_analysis)
    
    while True:
        schedule.run_pending()
        import time
        time.sleep(60)

def start_scheduler():
    """Iniciar thread do agendador"""
    scheduler_thread = threading.Thread(target=schedule_updates, daemon=True)
    scheduler_thread.start()
    print(f"📅 Agendador iniciado (atualização a cada {UPDATE_INTERVAL_MINUTES} minutos)")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MARKET SENTIMENT ANALYTICS - FLASK SERVER")
    print("="*60)
    
    # Executar análise inicial
    print("\n📊 Executando análise inicial...")
    try:
        report = analyzer.generate_report()
        analyzer.save_report(report, DATA_FILE)
        analyzer.print_summary(report)
    except Exception as e:
        print(f"⚠️  Erro na análise inicial: {e}")
    
    # Iniciar agendador
    start_scheduler()
    
    # Iniciar servidor
    port = int(os.getenv('FLASK_PORT', 5000))
    print(f"\n🌐 Servidor iniciado em http://localhost:{port}")
    print("   Acesse http://localhost:{}/")
    print("   Dashboard será atualizada automaticamente a cada {} minutos".format(port, UPDATE_INTERVAL_MINUTES))
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=port, use_reloader=False)