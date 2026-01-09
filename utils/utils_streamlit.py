import streamlit as st

def botao_mudar_tema():

    if 'themebutton' not in st.session_state:
        st.session_state['themebutton'] = 'light'  # Padrão: tema light

    # Define o ícone do botão com base no tema atual
    icon = "🌞" if st.session_state['themebutton'] == 'light' else "🌙"

    # Cria o botão para alternar entre temas
    if st.button(icon=icon, label="Mudar Tema", help="Alternar entre modo claro e escuro", key="switch_theme_button", use_container_width=False):
        
        if st.session_state['themebutton'] == 'light':
            st._config.set_option("theme.base", "dark")

            st.session_state['themebutton'] = 'dark'
        else:
            st._config.set_option("theme.base", "light")

            st.session_state['themebutton'] = 'light'

        st.rerun()  # Recarrega a interface para aplicar o novo tema