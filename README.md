# AgroSearch — Motor de Busca Inteligente

Protótipo de motor de busca textual para manuais técnicos de agricultura. O pipeline de pré-processamento, o **índice invertido**, o **TF-IDF** e a **similaridade de cosseno** foram implementados **do zero**, só com a biblioteca padrão do Python (sem scikit-learn, NLTK ou `TfidfVectorizer`). Streamlit e pandas são usados apenas para a interface.

## Entregáveis

| Arquivo | Conteúdo |
|---|---|
| [`agrosearch_app.py`](agrosearch_app.py) | Aplicação completa em Streamlit (arquivo único) |
| [`RELATORIO.pdf`](RELATORIO.pdf) | Relatório (2 páginas) com arquitetura, exemplo de cálculo e divisão de tarefas |

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run agrosearch_app.py
```

## O que o app faz

| Aba | Fase | Destaques |
|---|---|---|
| 1️⃣ Pré-processamento | Tokenização → normalização → stopwords → stemming | caixas de seleção para stopwords e stemming; resultado de cada etapa; tamanho do vocabulário nas 4 combinações |
| 2️⃣ Índice Invertido | termo → {documento: frequência} | exibido em `st.dataframe` (com df, postings e IDF) e em `st.json` |
| 3️⃣ Busca TF-IDF | TF, IDF = log₁₀(N/df), TF-IDF acumulado | ranking do maior para o menor, documento vencedor destacado, passo a passo do cálculo e matriz TF-IDF |
| ⭐ Similaridade de Cosseno | bônus | cosseno entre o vetor da consulta e os vetores TF-IDF, comparado com o ranking acumulado |

O stemmer (`StemmerPT`) é uma versão simplificada das etapas do RSLP: *irrigação, irrigações, irrigar* → `irrig`; *cultivo, cultivar* → `cultiv`. A base de documentos pode ser editada pela barra lateral, e o índice é reconstruído na hora.
