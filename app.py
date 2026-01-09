import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import dill
import json
import html
import warnings
warnings.filterwarnings('ignore')

from utils.utils_streamlit import botao_mudar_tema

st.set_page_config(page_title="Predição de Obesidade", page_icon="🩺", layout="wide")

# Classes consideradas "seguras" conforme tradução anterior
SAFE_CLASSES = {"Abaixo do Peso", "Peso Normal"}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering ajustada para nomes em português."""
    df = df.copy()

    # Cálculo do IMC (BMI) usando nomes traduzidos
    if {"Peso", "Altura"}.issubset(df.columns):
        df["IMC"] = df["Peso"] / (df["Altura"] ** 2)

    # Binning baseado no dicionário traduzido [cite: 9, 12, 17, 20, 22]
    bins_map = {
        "Consumo_Vegetais": ([0, 1.5, 2.5, 10], ["baixa", "moderada", "alta"]),
        "Numero_Refeicoes_Principais": ([0, 2.5, 3.5, 10], ["1-2", "3", "4+"]),
        "Consumo_Agua_Diario": ([0, 1.5, 2.5, 10], ["<1L", "1-2L", ">2L"]),
        "Atividade_Fisica_Frequencia": ([0, 0.5, 1.5, 2.5, 10], ["nenhuma", "baixa", "media", "alta"]),
        "Tempo_Dispositivos_Tecnologicos": ([0, 0.5, 1.5, 10], ["0-2h", "3-5h", ">5h"]),
    }
    
    for col, (bins, labels) in bins_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            cat = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=False)
            df[col] = cat.astype(str)
            
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].fillna("missing").astype(str)
    return df

# Necessário para carregar modelos que usam FunctionTransformer do __main__
sys.modules["__main__"].engineer_features = engineer_features


@st.cache_resource(show_spinner=False)
def load_artifact(path: Path):
    """Carrega o artefato persistido usando dill para evitar erro de __builtin__."""
    with open(path, "rb") as f:
        artifact = dill.load(f)
    
    # Retorna o pipeline e o label_encoder conforme salvo no dicionario
    if isinstance(artifact, dict):
        pipeline, label_encoder = artifact["pipeline"], artifact["label_encoder"]
    else:
        pipeline, label_encoder = artifact, None

    # Reaponta o FunctionTransformer para a implementacao local, evitando bytecode incompativel entre ambientes
    if hasattr(pipeline, "named_steps") and "feature_engineering" in pipeline.named_steps:
        from sklearn.preprocessing import FunctionTransformer
        fe_step = pipeline.named_steps["feature_engineering"]

        new_fe = FunctionTransformer(
            func=engineer_features,
            inverse_func=getattr(fe_step, "inverse_func", None),
            validate=getattr(fe_step, "validate", False),
            accept_sparse=getattr(fe_step, "accept_sparse", False),
            check_inverse=getattr(fe_step, "check_inverse", True),
            feature_names_out=getattr(fe_step, "feature_names_out", None),
            kw_args=getattr(fe_step, "kw_args", None),
            inv_kw_args=getattr(fe_step, "inv_kw_args", None),
        )

        try:
            pipeline.set_params(feature_engineering=new_fe)
        except Exception:
            pipeline.steps = [
                (name, new_fe if name == "feature_engineering" else step)
                for name, step in pipeline.steps
            ]
            if hasattr(pipeline, "named_steps"):
                pipeline.named_steps["feature_engineering"] = new_fe

    return pipeline, label_encoder

def _select_mapped(label: str, options: dict, default_key: str):
    labels = list(options.keys())
    idx = labels.index(default_key)
    choice = st.selectbox(label, labels, index=idx)
    return options[choice]

def collect_user_input():
    st.subheader("Informe seus dados:")
    nome_paciente = st.text_input("Nome do paciente (opcional)", placeholder="Digite para identificar o relatório")

    # Mapeamentos para o formato que o modelo espera (Português)
    fcvc_map = {"Raramente": 1.0, "Às vezes": 2.0, "Sempre": 3.0}
    ncp_map = {"1 refeição": 1.0, "2 refeições": 2.0, "3 refeições": 3.0, "4 ou mais": 4.0}
    ch2o_map = {"Menos de 1L/dia": 1.0, "1 a 2L/dia": 2.0, "Mais de 2L/dia": 3.0}
    faf_map = {
        "Nenhuma": 0.0, 
        "1 a 2 vezes por semana": 1.0, 
        "3 a 4 vezes por semana": 2.0, 
        "5 ou mais vezes por semana": 3.0
    }
    tue_map = {"0 a 2 horas/dia": 0.0, "3 a 5 horas/dia": 1.0, "Mais de 5 horas/dia": 2.0}
    
    # Dicionários já existentes
    sim_nao = {"Sim": "Sim", "Não": "Não"}
    genero_map = {"Feminino": "Feminino", "Masculino": "Masculino"}
    frequencia_map = {"Não": "Não", "Às vezes": "Às vezes", "Frequentemente": "Frequentemente", "Sempre": "Sempre"}
    mtrans_map = {
        "Transporte Público": "Transporte Público",
        "Caminhada": "Caminhada",
        "Automóvel": "Automóvel",
        "Motocicleta": "Motocicleta",
        "Bicicleta": "Bicicleta",
    }

    col1, col2 = st.columns(2)
    with col1:

        with st.expander('Dados iniciais', expanded=True):
            genero = _select_mapped("Gênero", genero_map, "Feminino")
            idade = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25, step=1)
            altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.00, value=1.70, step=0.01)
            peso = st.number_input("Peso (kg)", min_value=39.0, max_value=175.0, value=70.0, step=0.5)

            imc = None
            classe_imc = None

            if peso and altura:
                # Cálculo do IMC conforme a fórmula padrão
                imc = round(peso / (altura ** 2), 2)
                
                # Lógica de classificação baseada no dicionário e referências visuais
                if imc < 18.5:
                    classe_imc = "Abaixo do Peso"
                    cor_aviso = "info"
                elif 18.5 <= imc < 25:
                    classe_imc = "Peso Ideal (Normal)"
                    cor_aviso = "success"
                elif 25 <= imc < 30:
                    classe_imc = "Sobrepeso"
                    cor_aviso = "warning"
                elif 30 <= imc < 35:
                    classe_imc = "Obesidade Grau I"
                    cor_aviso = "error"
                elif 35 <= imc < 40:
                    classe_imc = "Obesidade Grau II"
                    cor_aviso = "error"
                else:
                    classe_imc = "Obesidade Grau III (Mórbida)"
                    cor_aviso = "error"

                # Exibição do aviso personalizado
                st.metric("Seu IMC Atual", imc)
                
                if cor_aviso == "success":
                    st.success(f"Classificação por IMC: **{classe_imc}**")
                elif cor_aviso == "info":
                    st.info(f"Classificação por IMC: **{classe_imc}**")
                elif cor_aviso == "warning":
                    st.warning(f"Classificação por IMC: **{classe_imc}**")
                else:
                    st.error(f"Classificação por IMC: **{classe_imc}**")

    with col2:
        with st.expander('Descreva brevemente sua dieta:', expanded=True):
            ncp = _select_mapped("Número de refeições principais por dia (NCP)", ncp_map, "3 refeições")
            caec = _select_mapped("Consumo de lanches entre refeições (CAEC)", frequencia_map, "Às vezes")
            fcvc = _select_mapped("Frequência de consumo de vegetais (FCVC)", fcvc_map, "Às vezes")
            ch2o = _select_mapped("Consumo diário de água (CH2O)", ch2o_map, "1 a 2L/dia")
            favc = _select_mapped("Consumo frequente de alimentos calóricos? (FAVC)", sim_nao, "Sim")
            calc = _select_mapped("Consumo de álcool (CALC)", frequencia_map, "Não")

    with st.expander('Como são seus hábitos?', expanded=True):
        smoke = _select_mapped("É fumante? (SMOKE)", sim_nao, "Não")
        scc = _select_mapped("Monitora calorias? (SCC)", sim_nao, "Não")
        faf = _select_mapped("Frequência de atividade física semanal (FAF)", faf_map, "1 a 2 vezes por semana")
        tue = _select_mapped("Tempo diário em dispositivos eletrônicos (TUE)", tue_map, "0 a 2 horas/dia")
        mtrans = _select_mapped("Meio de transporte principal", mtrans_map, "Transporte Público")
        hist_fam = _select_mapped("Histórico familiar de sobrepeso?", sim_nao, "Sim")

    data = {
    "Genero": genero, "Idade": float(idade), "Altura": float(altura), "Peso": float(peso),
    "Historico_Familiar_Sobrepeso": hist_fam, "Consumo_Calorico_Frequente": favc,
    "Consumo_Vegetais": float(fcvc), "Numero_Refeicoes_Principais": float(ncp),
    "Consumo_Alimentos_Entre_Refeicoes": caec, "Fumante": smoke,
    "Consumo_Agua_Diario": float(ch2o), "Monitoramento_Calorias": scc,
    "Atividade_Fisica_Frequencia": float(faf), "Tempo_Dispositivos_Tecnologicos": float(tue),
    "Consumo_Alcool": calc, "Meio_Transporte": mtrans
    }
    meta_paciente = {
        "nome": nome_paciente.strip() if nome_paciente else "",
        "idade": float(idade),
        "altura": float(altura),
        "peso": float(peso),
        "imc": imc,
        "classe_imc": classe_imc
    }
    return pd.DataFrame([data]), meta_paciente


def montar_relatorio_html(paciente_info, classe_final, prob_risco, prob_df: pd.DataFrame) -> str:
    """Gera um HTML simples para ser impresso ou salvo pelo paciente."""
    nome = paciente_info.get("nome") or "Paciente"
    nome = html.escape(nome)

    imc_valor = paciente_info.get("imc")
    classe_imc = paciente_info.get("classe_imc") or "-"

    prob_rows = []
    for _, row in prob_df.iterrows():
        prob_rows.append(
            f"<tr><td>{html.escape(str(row['Classe']))}</td><td>{row['Probabilidade']*100:.1f}%</td></tr>"
        )
    prob_table = "\n".join(prob_rows)

    imc_bloco = ""
    if imc_valor is not None:
        imc_bloco = f"""
            <div class='metric'>
                <div class='metric-label'>IMC</div>
                <div class='metric-value'>{imc_valor}</div>
                <div class='metric-hint'>{classe_imc}</div>
            </div>
        """

    relatorio_html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="utf-8">
        <title>Relatorio do Paciente</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
            h1 {{ margin-bottom: 4px; }}
            .sub {{ color: #6b7280; margin-bottom: 24px; }}
            .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
            .metric {{ background: #f3f4f6; padding: 12px 16px; border-radius: 8px; min-width: 160px; }}
            .metric-label {{ font-size: 12px; letter-spacing: 0.04em; color: #6b7280; text-transform: uppercase; }}
            .metric-value {{ font-size: 22px; font-weight: 700; }}
            .metric-hint {{ color: #111827; margin-top: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
            th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
            th {{ background: #f9fafb; font-size: 12px; letter-spacing: 0.02em; text-transform: uppercase; color: #6b7280; }}
        </style>
    </head>
    <body>
        <h1>Relatorio de Predição de Obesidade</h1>
        <div class="sub">Resumo gerado para {nome}</div>
        <div class="cards">
            <div class='metric'>
                <div class='metric-label'>Classe Prevista</div>
                <div class='metric-value'>{html.escape(str(classe_final))}</div>
            </div>
            <div class='metric'>
                <div class='metric-label'>Risco Total</div>
                <div class='metric-value'>{prob_risco*100:.1f}%</div>
                <div class='metric-hint'>Sobrepeso + Obesidade I/II/III</div>
            </div>
            {imc_bloco}
        </div>
        <h3 style="margin-top:24px;">Probabilidades por classe</h3>
        <table>
            <thead><tr><th>Classe</th><th>Probabilidade</th></tr></thead>
            <tbody>
                {prob_table}
            </tbody>
        </table>
    </body>
    </html>
    """
    return relatorio_html


def mostrar_controles_impressao(relatorio_html: str):
    """Mostra botões para baixar e imprimir o relatório do paciente."""

    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
        "Baixar relatório (HTML)",
        data=relatorio_html.encode("utf-8"),
        file_name="relatorio_paciente.html",
        mime="text/html",
        help="Abra o arquivo baixado e use Ctrl+P para imprimir ou salvar em PDF."
        )

    with col2:
        components.html(
            f"""
            <div style="margin-top: 8px;">
                <button style="padding: 8px 14px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer;" onclick="printReport()">Imprimir relatório agora</button>
            </div>
            <script>
                const report = {json.dumps(relatorio_html)};
                function printReport() {{
                    const w = window.open("", "RelatorioPaciente", "width=900,height=650");
                    if (!w) return;
                    w.document.write(report);
                    w.document.close();
                    w.focus();
                    w.print();
                }}
            </script>
            """,
            height=110
        )


def main():

    botao_mudar_tema()


    st.title("🩺 :blue[Sistema de Predição de Nível de Obesidade]")
    st.write(":blue[FIAP - Tech Challenge Fase 4]")

    tab1, tab2, tab3 = st.tabs([
        'Início',
        'Predição',
        'Painel Analítico'
    ])

    with tab1:
        st.markdown(
            """
            ### :blue[Objetivo do Projeto]
            Desenvolver um modelo de Machine Learning para auxiliar os médicos e médicas a prever se uma pessoa pode ter obesidade.

            ### :blue[Metodologia (notebooks e escolha do modelo)]
            - Exploração e limpeza dos dados em notebook: análise de distribuição das variáveis, tratamento de valores faltantes e criação do IMC.
            - Engenharia de atributos: tratamento de variáveis contínuas para categorias interpretáveis e normalização de nomes para o idioma português.
            - Treinamento/validação: comparação de RandomForest (com/sem SMOTE), LightGBM e XGBoost ([acesse o notebook](https://link.com)), medindo acurácia, F1-macro e recall para classes de risco. O XGBoost liderou em acurácia e recall de risco, mantendo desempenho parecido entre treino e validação (sem overfitting).
            - Seleção do modelo: XGBoost (Acurácia: 96,2% - F1-macro: 96,1% - Recall risco: 99,36%) serializado (pipeline + label encoder) e carregado pelo app para predições em produção.

            ### :blue[A solução em produção (Streamlit)]
            - Coleta guiada dos dados do paciente com cálculo do IMC e faixa de risco.
            - Predição via pipeline treinado e label encoder carregados dos artefatos persistidos.
            - Exibição de probabilidades por classe com gráfico de barras percentuais, além do risco acumulado.
            - Geração de relatório HTML para download ou impressão (ex.: PDF) direto pelo usuário.

            ### :blue[Equipe responsável]
            - Alisson Cordeiro Nóbrega
            - Lucas Benevides Miranda
            - Marcos Vinícius Fernandes de Freitas
            - Rodrigo Mallet e Ribeiro de Carvalho
            """
        )
        st.divider()

    with tab2:


        model_path = Path("models/xgboost.pkl")

        if not model_path.exists():
            st.error(f"Modelo não encontrado em {model_path}")
            st.stop()

        pipeline, label_encoder = load_artifact(model_path)

        with st.expander('Informações', expanded=True):
            input_df, meta_paciente = collect_user_input()

        if st.button("Calcular Probabilidade", type='primary'):

            st.toast('Relatório para download disponível.', icon='🖨️', duration='short')


            # Realiza a predição
            probs = pipeline.predict_proba(input_df)[0]
            classes = label_encoder.classes_
            
            # Identifica a classe prevista
            idx_pred = np.argmax(probs)
            classe_final = classes[idx_pred]
            
            # Separa risco de obesidade (Sobrepeso + Obesidade)
            classes_risco = [c for c in classes if c not in SAFE_CLASSES]
            prob_risco = sum(probs[i] for i, c in enumerate(classes) if c in classes_risco)
            prob_df = pd.DataFrame({"Classe": classes, "Probabilidade": probs}).sort_values("Probabilidade", ascending=False)

            # Exibição de resultados
            c1, c2, c3= st.columns(3)
            with c1:
                st.success(f"### Resultado: **{classe_final}**")
            with c2:
                cor_risco = "inverse" if prob_risco > 0.5 else "normal"
                st.metric("Risco Total Acumulado", f"{prob_risco*100:.1f}%", help="Soma das probabilidades de Sobrepeso e Obesidade I, II e III",
                          width='stretch')
            with c3:
                st.subheader("Relatório para o paciente")
                relatorio_html = montar_relatorio_html(meta_paciente, classe_final, prob_risco, prob_df)
                mostrar_controles_impressao(relatorio_html)
            
            st.subheader("Distribuição de Probabilidade por Classe de peso corporal")
            prob_df_vis = prob_df.assign(
                Probabilidade_pct=lambda df: df["Probabilidade"] * 100,
                Probabilidade_label=lambda df: df["Probabilidade"].mul(100).map(lambda v: f"{v:.1f}%")
            )

            tema_atual = st.session_state.get("themebutton", "light")
            label_color = "#0f172a" if tema_atual != "dark" else "#f8fafc"

            bar_height = max(260, 70 * len(prob_df_vis))
            bars = (
                alt.Chart(prob_df_vis)
                .mark_bar(size=48, cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    y=alt.Y("Classe:N", sort="-x", title="Classe"),
                    x=alt.X("Probabilidade_pct:Q", title="Probabilidade (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.value("#2563eb"),
                    tooltip=[
                        alt.Tooltip("Classe:N", title="Classe"),
                        alt.Tooltip("Probabilidade_pct:Q", title="Probabilidade (%)", format=".1f"),
                    ],
                )
                .properties(height=bar_height)
            )

            labels = (
                alt.Chart(prob_df_vis)
                .mark_text(align="left", baseline="middle", dx=10, fontSize=13, color=label_color)
                .encode(
                    y=alt.Y("Classe:N", sort="-x"),
                    x=alt.X("Probabilidade_pct:Q"),
                    text="Probabilidade_label:N",
                )
            )

            st.altair_chart(bars + labels, use_container_width=True)


    with tab3:

        path_bi = "https://app.powerbi.com/view?r=eyJrIjoiN2M4NDc5MjItZWFmYS00ZDA3LTk3ODYtZTk0MWEwM2Y5Njg5IiwidCI6ImQ3M2IwYzU2LWU5NGYtNDMzNi04YjBjLTdhMTY3NThhMWIzOCJ9&pageName=af4c47019749392c0a0a"

        st.markdown(
            f"<div style='text-align: center;'>"
            f"<iframe src='{path_bi}' "
            f"width='1280' height='720' "
            f"frameborder='0' allowfullscreen='true' style='width: 100%; height: 80vh;'></iframe>"
            f"</div>",
            unsafe_allow_html=True
        )



if __name__ == "__main__":
    main()
