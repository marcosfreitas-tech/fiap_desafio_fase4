# Documentação Geral do Projeto

## Visão Geral
- **Objetivo**: prever o nível de obesidade a partir de dados clínicos e de estilo de vida, oferecendo feedback rápido e um relatório imprimível para pacientes.
- **Principais componentes**: notebook de análise/modelagem (`analise/model_comparison.ipynb`), app Streamlit (`app.py`), artefato do modelo (`models/xgboost.pkl`), utilitário de tema (`utils/utils_streamlit.py`) e dados de apoio em `data/`.

## Estrutura do Repositório
- `app.py`: aplicação Streamlit com três abas (Início, Predição, Painel Analítico). Coleta dados do paciente, calcula IMC, gera predição, mostra gráfico de probabilidades e oferece download/impressão do relatório em HTML.
- `utils/utils_streamlit.py`: botão de alternância de tema claro/escuro usando `st.session_state` e `st._config` para aplicar o tema.
- `analise/model_comparison.ipynb`: notebook de exploração, engenharia de atributos e comparação de modelos (RandomForest com/sem SMOTE, LightGBM, XGBoost). Base para a escolha do modelo final.
- `models/xgboost.pkl`: artefato serializado com `dill` contendo pipeline de pré-processamento + classificador XGBoost + label encoder.
- `data/`: insumos de dados (ex.: CSV de obesidade, dicionário de atributos).
- `doc/`: documentação (PDFs fornecidos e este guia).
- `.streamlit/config.toml`: ajustes de layout/tema do Streamlit.
- `requirements.txt`: dependências da aplicação (pandas, scikit-learn, xgboost, streamlit, altair, etc.).
- `.gitignore`: arquivos ignorados no controle de versão.

## Como Rodar
1) Criar ambiente e instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```
2) Executar a aplicação web:
   ```bash
   streamlit run app.py
   ```
3) Abrir o endereço indicado pelo Streamlit no navegador (geralmente http://localhost:8501).

## Como Usar o App
- **Início**: resumo do projeto, metodologia e equipe.
- **Predição**: informar dados do paciente (dados iniciais, dieta, hábitos), calcular IMC automático, gerar predição e risco total, visualizar gráfico de probabilidades e baixar/imprimir relatório HTML.
- **Painel Analítico**: incorpora visualização externa (Power BI) para exploração adicional.

## Fluxo de Predição (app.py)
1) **Entrada**: campos numéricos e categóricos guiados; cálculo de IMC com feedback de faixa de risco.
2) **Pré-processamento**: função `engineer_features` (cálculo de IMC, binning de colunas contínuas, normalização de texto) e label encoding carregado do artefato.
3) **Modelo**: pipeline XGBoost carregado de `models/xgboost.pkl` (serializado com `dill`).
4) **Saída**: classe prevista, risco acumulado de sobrepeso/obesidade, gráfico horizontal de probabilidades (Altair) com rótulos percentuais e tema responsivo, relatório HTML para download/impressão.

## Referências de Documentos
- `doc/dicionario_obesity_fiap.pdf`: dicionário de atributos do dataset.
- `doc/POSTECH - Tech Challenge - Fase 4 - Data Analytics_ (1).pdf`: material do desafio.

## Próximos Passos Sugeridos
- Acrescentar testes automatizados para o pipeline de inferência (ex.: entradas mínimas e máximas).
- Adicionar monitoramento de métricas em produção (logging das distribuições de entrada e das saídas de classe).-
