"""
Market Sentiment Analyzer - Backend
Análise de sentimento de mercado em tempo real com NEWS API e Transformers
"""

import requests
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
from transformers import pipeline
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore')

# Carregar variáveis de ambiente
load_dotenv()

class MarketSentimentAnalyzer:
    """Analisador de sentimento do mercado"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = 'https://newsapi.org/v2/everything'
        
        # Categorias de ativos
        self.asset_categories = {
            'techs': [
                'Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 
                'NVIDIA', 'Meta', 'Intel', 'AMD', 'Qualcomm',
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'tech', 'technology', 'smartphone', 'cloud',
                'artificial intelligence', 'AI', 'machine learning'
            ],
            'criptos': [
                'Bitcoin', 'Ethereum', 'BTC', 'ETH', 'Cardano', 'ADA',
                'Solana', 'SOL', 'Polkadot', 'DOT', 'Ripple', 'XRP',
                'Dogecoin', 'DOGE', 'cryptocurrency', 'crypto', 'blockchain',
                'DeFi', 'NFT', 'Web3', 'token', 'altcoin', 'Bitcoin Cash',
                'Litecoin', 'LTC', 'Monero', 'XMR'
            ],
            'gold': [
                'ouro', 'gold', 'GLD', 'aurum', 'precious metals',
                'ouro spot', 'bullion', 'onça de ouro', 'troy ounce gold',
                'commodity oro', 'mercado de ouro', 'preço do ouro',
                'gold price', 'gold market', 'investimento ouro',
                'GOLDBEES', 'EFT ouro'
            ],
            'silver': [
                'prata', 'silver', 'SLV', 'argentum', 'prata spot',
                'prata bullion', 'onça de prata', 'troy ounce silver',
                'commodity plata', 'mercado de prata', 'preço da prata',
                'silver price', 'silver market', 'investimento prata',
                'SILVERBEES'
            ],
            'energias_renovaveis': [
                'NextEra', 'solar', 'eólica', 'renewable energy'
            ]
        }
        
        # Pipeline de análise de sentimento
        self.sentiment_pipeline = None
        self._initialize_sentiment_model()
        
        # Armazenamento de dados
        self.market_data = defaultdict(lambda: defaultdict(list))
        self.articles_by_asset = defaultdict(list)
    
    def _initialize_sentiment_model(self):
        """Inicializa o modelo de análise de sentimento"""
        print("🤖 Carregando modelo de sentimento (DistilBERT)...")
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU (mude para 0 se tiver GPU)
            )
            print("✅ Modelo carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
    def fetch_news(self, keywords, days=1):
        """Busca notícias via NEWS API"""
        print(f"📰 Buscando notícias sobre: {keywords}")
        
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'q': keywords,
            'from': from_date,
            'to': to_date,
            'sortBy': 'relevancy',
            'pageSize': 100,
            'apiKey': self.api_key,
            'language': 'en'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            print(f"   → Encontradas {len(articles)} notícias")
            
            return articles
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao buscar notícias: {e}")
            return []
    
    def analyze_sentiment(self, text, max_length=512):
        """Analisa sentimento de um texto"""
        if not self.sentiment_pipeline or not text:
            return None
        
        try:
            # Limitar tamanho do texto
            text = text[:max_length]
            
            result = self.sentiment_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Mapear POSITIVE/NEGATIVE para escala -1 a 1
            sentiment = {
                'POSITIVE': (1, score),
                'NEGATIVE': (-1, score)
            }.get(label, (0, 0.5))
            
            return sentiment
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return (0, 0.5)
    
    def process_articles(self, articles, category):
        """Processa artigos e analisa sentimentos"""
        sentiments = []
        processed_count = 0
        
        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')
                source = article.get('source', {}).get('name', 'Unknown')
                
                # Combinar título e descrição
                text = f"{title} {description}"
                
                # Analisar sentimento
                sentiment_result = self.analyze_sentiment(text)
                
                if sentiment_result:
                    sentiment_value, confidence = sentiment_result
                    
                    sentiment_data = {
                        'title': title,
                        'url': url,
                        'source': source,
                        'sentiment': sentiment_value,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    sentiments.append(sentiment_data)
                    processed_count += 1
                    
                    # Mostrar progresso
                    if processed_count % 10 == 0:
                        print(f"   → Processados {processed_count} artigos...")
            
            except Exception as e:
                print(f"   ⚠️ Erro ao processar artigo: {e}")
                continue
        
        return sentiments
    
    def categorize_and_analyze(self):
        """Categoriza ativos e analisa sentimentos por categoria"""
        print("\n📊 Iniciando análise de categorias...\n")
        
        results = {}
        
        for category, assets in self.asset_categories.items():
            print(f"🔍 Analisando categoria: {category.upper()}")
            
            # Buscar notícias para cada ativo da categoria
            all_articles = []
            for asset in assets:
                articles = self.fetch_news(asset, days=1)
                all_articles.extend(articles)
            
            # Remover duplicatas
            unique_articles = {a['url']: a for a in all_articles}.values()
            
            # Processar e analisar
            sentiments = self.process_articles(list(unique_articles), category)
            
            if sentiments:
                # Calcular estatísticas
                df = pd.DataFrame(sentiments)
                
                positive = len(df[df['sentiment'] > 0])
                negative = len(df[df['sentiment'] < 0])
                neutral = len(df[df['sentiment'] == 0])
                total = len(df)
                
                results[category] = {
                    'positive_count': int(positive),
                    'positive_pct': round((positive / total * 100) if total > 0 else 0, 2),
                    'negative_count': int(negative),
                    'negative_pct': round((negative / total * 100) if total > 0 else 0, 2),
                    'neutral_count': int(neutral),
                    'neutral_pct': round((neutral / total * 100) if total > 0 else 0, 2),
                    'total_mentions': total,
                    'avg_confidence': round(df['confidence'].mean(), 3),
                    'articles': sentiments[:10]  # Top 10 artigos
                }
                
                print(f"   ✅ {category}: {positive}+ / {negative}- / {neutral}= (total: {total})\n")
            else:
                results[category] = {
                    'positive_count': 0,
                    'positive_pct': 0,
                    'negative_count': 0,
                    'negative_pct': 0,
                    'neutral_count': 0,
                    'neutral_pct': 0,
                    'total_mentions': 0,
                    'avg_confidence': 0,
                    'articles': []
                }
        
        return results
    
    def identify_top_assets(self, all_data):
        """Identifica ativos mais/menos falados e com melhor/pior sentimento"""
        asset_stats = defaultdict(lambda: {
            'mentions': 0,
            'positive': 0,
            'negative': 0,
            'sentiment_score': 0
        })
        
        # Coletar dados de todos os artigos
        for category, data in all_data.items():
            for article in data.get('articles', []):
                # Extrair ativos mencionados do título
                for asset_list in self.asset_categories.values():
                    for asset in asset_list:
                        if asset.lower() in article['title'].lower():
                            asset_stats[asset]['mentions'] += 1
                            
                            if article['sentiment'] > 0:
                                asset_stats[asset]['positive'] += 1
                            elif article['sentiment'] < 0:
                                asset_stats[asset]['negative'] += 1
                            
                            asset_stats[asset]['sentiment_score'] += article['sentiment']
        
        # Converter para lista
        assets_list = [
            {
                'asset': asset,
                'mentions': stats['mentions'],
                'positive': stats['positive'],
                'negative': stats['negative'],
                'sentiment_avg': round(
                    stats['sentiment_score'] / stats['mentions'], 3
                ) if stats['mentions'] > 0 else 0
            }
            for asset, stats in asset_stats.items()
            if stats['mentions'] > 0
        ]
        
        return {
            'most_talked': sorted(
                assets_list,
                key=lambda x: x['mentions'],
                reverse=True
            )[:10],
            'least_talked': sorted(
                assets_list,
                key=lambda x: x['mentions']
            )[:10],
            'most_positive': sorted(
                assets_list,
                key=lambda x: x['sentiment_avg'],
                reverse=True
            )[:10],
            'most_negative': sorted(
                assets_list,
                key=lambda x: x['sentiment_avg']
            )[:10]
        }
    
    def generate_report(self):
        """Gera relatório completo de análise"""
        print("🚀 Iniciando análise de sentimento do mercado...\n")
        
        # Analisar categorias
        category_results = self.categorize_and_analyze()
        
        # Identificar top assets
        top_assets = self.identify_top_assets(category_results)
        
        # Compilar resultado final
        report = {
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': {
                category: {
                    'positive': data['positive_pct'],
                    'negative': data['negative_pct'],
                    'neutral': data['neutral_pct'],
                    'total_mentions': data['total_mentions'],
                    'avg_confidence': data['avg_confidence'],
                    'articles_count': data['positive_count'] + data['negative_count'] + data['neutral_count']
                }
                for category, data in category_results.items()
            },
            'top_assets': top_assets,
            'detailed_category_data': category_results
        }
        
        return report
    
    def save_report(self, report, filename='market_sentiment_data.json'):
        """Salva relatório em JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Relatório salvo em: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Erro ao salvar relatório: {e}")
            return None
    
    def print_summary(self, report):
        """Exibe resumo da análise"""
        print("\n" + "="*60)
        print("📊 RESUMO DA ANÁLISE DE SENTIMENTO DO MERCADO")
        print("="*60)
        
        print(f"\n⏰ Timestamp: {report['timestamp']}")
        
        print("\n📈 SENTIMENTO POR CATEGORIA:")
        print("-" * 60)
        
        for category, sentiment in report['market_sentiment'].items():
            print(f"\n{category.upper()}:")
            print(f"  ✅ Positivo: {sentiment['positive']}%")
            print(f"  ❌ Negativo: {sentiment['negative']}%")
            print(f"  ⚪ Neutro:   {sentiment['neutral']}%")
            print(f"  📰 Total de menções: {sentiment['total_mentions']}")
        
        print("\n🔥 TOP ASSETS MAIS FALADOS:")
        print("-" * 60)
        for i, asset in enumerate(report['top_assets']['most_talked'][:5], 1):
            print(f"{i}. {asset['asset']}: {asset['mentions']} menções")
        
        print("\n💎 SENTIMENTO MAIS POSITIVO:")
        print("-" * 60)
        for i, asset in enumerate(report['top_assets']['most_positive'][:5], 1):
            print(f"{i}. {asset['asset']}: {asset['sentiment_avg']:.2%}")
        
        print("\n⚠️  SENTIMENTO MAIS NEGATIVO:")
        print("-" * 60)
        for i, asset in enumerate(report['top_assets']['most_negative'][:5], 1):
            print(f"{i}. {asset['asset']}: {asset['sentiment_avg']:.2%}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Função principal"""
    analyzer = MarketSentimentAnalyzer()
    
    # Gerar relatório
    report = analyzer.generate_report()
    
    # Salvar relatório
    analyzer.save_report(report)
    
    # Exibir resumo
    analyzer.print_summary(report)


if __name__ == '__main__':
    main()