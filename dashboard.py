import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Regressão Linear",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título do Dashboard
st.title("📈 Dashboard de Regressão Linear para Análise de Emissões de CO2")

# Carregando os dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("co2.csv")  
        st.success("Dados carregados com sucesso!")
    except FileNotFoundError:
        st.error("Arquivo 'co2.csv' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

    # Normalizar nomes das colunas: remover espaços extras e padronizar maiúsculas/minúsculas
    df.columns = df.columns.str.strip().str.capitalize()

    # Verificar tipos de dados e converter colunas numéricas se necessário
    numeric_columns = [
        'Engine size(l)', 'Cylinders', 
        'Fuel consumption city (l/100 km)', 
        'Fuel consumption hwy (l/100 km)', 
        'Fuel consumption comb (l/100 km)', 
        'Fuel consumption comb (mpg)', 
        'Co2 emissions(g/km)'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.warning(f"A coluna '{col}' não foi encontrada no DataFrame.")

    return df

df = load_data()

if df is not None:
    # Verificar se a coluna 'Co2 Emissions(g/km)' está presente
    target_column = 'Co2 emissions(g/km)'
    if target_column not in df.columns:
        st.error(f"A coluna '{target_column}' não foi encontrada nos dados. Verifique os nomes das colunas.")
        st.stop()

    # Selecionar apenas colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remover a coluna alvo de 'feature_options' de forma segura
    if target_column in numeric_columns:
        feature_options = numeric_columns.copy()
        feature_options.remove(target_column)
    else:
        feature_options = numeric_columns.copy()
        st.warning(f"A coluna '{target_column}' não está presente nas colunas numéricas. Certifique-se de que está correta.")

    # Identificar colunas categóricas
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Sidebar - Configurações
    st.sidebar.header("Configurações do Dashboard")

    # Adicionar uma seção de descrição das features na sidebar
    with st.sidebar.expander("🔍 Descrição das Features"):
        st.markdown("### Descrição das Features")
        st.markdown("""
        - **Make:** Marca do veículo.
        - **Model:** Modelo do veículo.
        - **Vehicle Class:** Classe do veículo.
        - **Engine Size(L):** Tamanho do motor em litros.
        - **Cylinders:** Número de cilindros do veículo.
        - **Transmission:** Tipo de transmissão.
        - **Fuel Type:** Tipo de combustível.
        - **Fuel Consumption City (L/100 km):** Consumo de combustível na cidade (L/100 km).
        - **Fuel Consumption Hwy (L/100 km):** Consumo de combustível na estrada (L/100 km).
        - **Fuel Consumption Comb (L/100 km):** Consumo combinado de combustível (L/100 km).
        - **Fuel Consumption Comb (mpg):** Consumo combinado de combustível (milhas por galão).
        - **Co2 Emissions(g/km):** Emissões de CO2 do veículo (g/km).
        """)

    # Opcional: Verificar os nomes das colunas
    with st.sidebar.expander("🔍 Verificar Colunas do DataFrame"):
        st.write("### Colunas Disponíveis:")
        st.write(df.columns.tolist())

    # Seleção de Abas
    tabs = st.tabs(["Visualizações", "Modelagem", "Previsões"])

    # 1. Aba de Visualizações
    with tabs[0]:
        st.subheader("Visualizações de Dados")

        # Seleção de Visualizações
        viz_options = st.multiselect(
            "Escolha as visualizações que deseja ver:",
            options=['Mapa de Correlação', 'Distribuição', 'Gráfico de Dispersão', 'Box Plot', 'Pair Plot', 'Heatmap'],
            default=['Mapa de Correlação', 'Distribuição']
        )

        # Utilizar expander para cada tipo de visualização
        if 'Mapa de Correlação' in viz_options:
            with st.expander("Mapa de Correlação"):
                st.write("### Mapa de Correlação")
                # Selecionar apenas colunas numéricas para correlação
                corr = df[numeric_columns].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Heatmap de Correlação")
                st.pyplot(fig)

        if 'Distribuição' in viz_options:
            with st.expander("Distribuição das Variáveis"):
                st.write("### Distribuição das Variáveis")
                selected_columns = st.multiselect(
                    "Selecione as colunas para ver a distribuição:",
                    options=numeric_columns,
                    default=numeric_columns
                )
                if selected_columns:
                    cols = st.columns(2)  # Distribuir histogramas em duas colunas
                    for idx, col in enumerate(selected_columns):
                        with cols[idx % 2]:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
                            ax.set_title(f'Distribuição de {col}')
                            st.pyplot(fig)
                else:
                    st.warning("Selecione pelo menos uma coluna para visualizar a distribuição.")

        if 'Gráfico de Dispersão' in viz_options:
            with st.expander("Gráfico de Dispersão"):
                st.write("### Gráfico de Dispersão")
                scatter_x = st.selectbox("Eixo X:", options=feature_options, index=0)
                scatter_y = st.selectbox("Eixo Y:", options=feature_options, index=1)
                hue_option = st.selectbox("Hue (Cor por):", options=['None'] + categorical_columns, index=0)

                fig, ax = plt.subplots(figsize=(8,6))
                if hue_option != 'None':
                    if hue_option in df.columns:
                        sns.scatterplot(data=df, x=scatter_x, y=scatter_y, hue=hue_option, palette='viridis', ax=ax, s=100, alpha=0.7)
                        ax.legend(title=hue_option, bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        st.warning(f"A coluna '{hue_option}' não foi encontrada no DataFrame.")
                        sns.scatterplot(data=df, x=scatter_x, y=scatter_y, color='blue', ax=ax, s=100, alpha=0.7)
                else:
                    sns.scatterplot(data=df, x=scatter_x, y=scatter_y, color='blue', ax=ax, s=100, alpha=0.7)
                ax.set_title(f'Scatter Plot de {scatter_x} vs {scatter_y}')
                st.pyplot(fig)

        if 'Box Plot' in viz_options:
            with st.expander("Box Plot das Variáveis"):
                st.write("### Box Plot das Variáveis")
                box_columns = st.multiselect(
                    "Selecione as colunas para o Box Plot:",
                    options=feature_options,
                    default=feature_options[:2] if len(feature_options) >=2 else feature_options
                )
                if box_columns:
                    cols = st.columns(2)  # Distribuir box plots em duas colunas
                    for idx, col in enumerate(box_columns):
                        with cols[idx % 2]:
                            fig, ax = plt.subplots(figsize=(8,4))
                            sns.boxplot(y=df[col], ax=ax, color='lightgreen')
                            ax.set_title(f'Box Plot de {col}')
                            st.pyplot(fig)
                else:
                    st.warning("Selecione pelo menos uma coluna para visualizar o Box Plot.")

        if 'Pair Plot' in viz_options:
            with st.expander("Pair Plot"):
                st.write("### Pair Plot")
                pair_columns = st.multiselect(
                    "Selecione até 5 colunas para o Pair Plot:",
                    options=feature_options + [target_column],
                    default=feature_options[:3] + [target_column] if len(feature_options) >=3 else feature_options + [target_column],
                    max_selections=5
                )
                if pair_columns:
                    hue_option = st.selectbox("Selecione a coluna para o Hue:", options=['None'] + categorical_columns, index=0)
                    if hue_option != 'None':
                        if hue_option in df.columns:
                            if hue_option not in pair_columns:
                                data_for_pairplot = df[pair_columns + [hue_option]]
                            else:
                                data_for_pairplot = df[pair_columns]
                        else:
                            st.warning(f"A coluna '{hue_option}' não foi encontrada no DataFrame.")
                            data_for_pairplot = df[pair_columns]
                    else:
                        data_for_pairplot = df[pair_columns]

                    try:
                        if hue_option != 'None':
                            fig = sns.pairplot(data_for_pairplot, hue=hue_option, palette='viridis', diag_kind='kde')
                        else:
                            fig = sns.pairplot(data_for_pairplot, diag_kind='kde')
                        st.pyplot(fig)
                    except KeyError as e:
                        st.error(f"Erro na criação do Pair Plot: {e}")
                else:
                    st.warning("Selecione pelo menos duas colunas para o Pair Plot.")

        if 'Heatmap' in viz_options:
            with st.expander("Heatmap de Features Específicas"):
                st.write("### Heatmap de Features Específicas")
                heatmap_columns = st.multiselect(
                    "Selecione as colunas para o Heatmap:",
                    options=feature_options,
                    default=feature_options[:2] if len(feature_options) >=2 else feature_options
                )
                if heatmap_columns:
                    # Incluir a coluna alvo se selecionada
                    cols_to_corr = heatmap_columns.copy()
                    if target_column not in cols_to_corr:
                        cols_to_corr.append(target_column)
                    corr_specific = df[cols_to_corr].corr()
                    fig, ax = plt.subplots(figsize=(10,8))
                    sns.heatmap(corr_specific, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title("Heatmap de Correlação entre Features Selecionadas")
                    st.pyplot(fig)
                else:
                    st.warning("Selecione pelo menos duas colunas para visualizar o Heatmap.")

    # 2. Aba de Modelagem
    with tabs[1]:
        st.subheader("Modelagem de Regressão Linear")

        # Seleção de Variáveis para Regressão
        selected_features = st.multiselect(
            "Selecione as variáveis independentes para a regressão:",
            options=feature_options,
            default=feature_options[:2] if len(feature_options) >=2 else feature_options
        )

        # Divisão dos dados e Treinamento do Modelo
        if len(selected_features) == 0:
            st.warning("Por favor, selecione pelo menos uma variável independente para a regressão.")
        else:
            X = df[selected_features]
            y = df[target_column]

            # Dividir os dados
            test_size = st.slider("Tamanho do conjunto de teste (%):", min_value=10, max_value=50, value=20, step=5)
            random_state = st.number_input("Random State:", value=42, step=1)

            # Botão para treinar o modelo
            if st.button("Treinar Modelo"):
                # Dividir os dados
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

                # Treinar o modelo
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Armazenar o modelo no session_state
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = model.predict(X_test)

                # Previsões
                y_pred = st.session_state['y_pred']
                y_test = st.session_state['y_test']

                # Métricas de Desempenho
                st.markdown("### Métricas de Desempenho")
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                metrics_df = pd.DataFrame({
                    'Métrica': ['MSE', 'RMSE', 'MAE', 'R²', 'R² (CV Média)', 'R² (CV Desvio)'],
                    'Valor': [mse, rmse, mae, r2, cv_scores.mean(), cv_scores.std()]
                })
                st.table(metrics_df)

                # Exibir Coeficientes do Modelo
                st.markdown("### Coeficientes do Modelo")
                coef_df = pd.DataFrame({
                    'Variável': selected_features,
                    'Coeficiente': model.coef_
                })
                coef_df.loc[len(coef_df)] = ['Intercepto', model.intercept_]
                st.table(coef_df)

                # Visualização das Previsões vs Valores Reais
                st.markdown("### Previsões vs Valores Reais")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='purple', s=100, alpha=0.7)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                ax.set_xlabel("Valores Reais")
                ax.set_ylabel("Previsões")
                ax.set_title("Previsões vs Valores Reais")
                st.pyplot(fig)

                # Resíduos
                st.markdown("### Análise dos Resíduos")
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8,6))
                sns.histplot(residuals, kde=True, ax=ax, color='orange')
                ax.set_title("Distribuição dos Resíduos")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(8,6))
                sns.scatterplot(x=y_pred, y=residuals, ax=ax, color='green', s=100, alpha=0.7)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_xlabel("Previsões")
                ax.set_ylabel("Resíduos")
                ax.set_title("Resíduos vs Previsões")
                st.pyplot(fig)

                # Exportar Modelo e Métricas
                st.markdown("### Exportar Resultados")
                # Converter coeficientes e métricas para CSV
                coef_csv = coef_df.to_csv(index=False).encode('utf-8')
                metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')

                # Botões de Download
                st.download_button(
                    label="📥 Baixar Coeficientes do Modelo (CSV)",
                    data=coef_csv,
                    file_name='coeficientes_modelo.csv',
                    mime='text/csv',
                )
                st.download_button(
                    label="📥 Baixar Métricas de Desempenho (CSV)",
                    data=metrics_csv,
                    file_name='metricas_desempenho.csv',
                    mime='text/csv',
                )

    # 3. Aba de Previsões
    with tabs[2]:
        st.subheader("Previsões Personalizadas")

        if len(selected_features) == 0:
            st.warning("Selecione variáveis independentes na aba 'Modelagem' para fazer previsões personalizadas.")
        else:
            st.markdown("### Insira os valores das variáveis para obter uma previsão de emissões de CO2:")

            # Verificar se o modelo foi treinado
            if 'model' not in st.session_state:
                st.error("Por favor, treine o modelo na aba 'Modelagem' antes de fazer previsões.")
            else:
                model = st.session_state['model']

                # Criação dos inputs para cada feature selecionada
                input_data = {}
                for feature in selected_features:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    step_val = (max_val - min_val) / 100 if (max_val - min_val) !=0 else 0.1
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step_val
                    )

                # Botão para realizar a previsão
                if st.button("Prever Emissões de CO2"):
                    input_df = pd.DataFrame([input_data])
                    try:
                        prediction = model.predict(input_df)[0]
                        st.success(f"As emissões de CO2 previstas são: **{prediction:.2f} g/km**")

                        # Adicionar à tabela de previsões para download
                        if 'historico_previsoes' not in st.session_state:
                            st.session_state.historico_previsoes = pd.DataFrame(columns=selected_features + [f"{target_column} Previsto"])

                        new_entry = input_df.copy()
                        new_entry[f"{target_column} Previsto"] = prediction
                        st.session_state.historico_previsoes = pd.concat([st.session_state.historico_previsoes, new_entry], ignore_index=True)

                        st.markdown("### Previsão Adicional")
                        st.dataframe(new_entry)
                    except Exception as e:
                        st.error(f"Erro na previsão: {e}")

            # Histórico de Previsões
            st.markdown("### Histórico de Previsões")
            if 'historico_previsoes' in st.session_state and not st.session_state.historico_previsoes.empty:
                # Exibir histórico em colunas para melhor organização
                cols = st.columns(2)
                with cols[0]:
                    st.dataframe(st.session_state.historico_previsoes)
                with cols[1]:
                    # Botão para baixar o histórico
                    csv_history = st.session_state.historico_previsoes.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Histórico de Previsões (CSV)",
                        data=csv_history,
                        file_name='historico_previsoes.csv',
                        mime='text/csv',
                    )

                # Opção para limpar o histórico
                if st.button("🔄 Limpar Histórico de Previsões"):
                    if 'historico_previsoes' in st.session_state:
                        st.session_state.historico_previsoes = pd.DataFrame(columns=selected_features + [f"{target_column} Previsto"])
                        st.success("Histórico de previsões limpo.")
            else:
                st.info("Nenhuma previsão realizada ainda.")
