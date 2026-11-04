import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
from scipy.sparse import csr_matrix, hstack

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Gravedad de Accidentes RUNT",
    page_icon="🚗",
    layout="centered"
)

# Ruta de la carpeta donde están guardados los artefactos del modelo
CARPETA = 'resultados_u4a1'

@st.cache_resource
def cargar_recursos():
    """Carga y cachea los estimadores y codificadores del modelo entrenado."""
    scaler = load(os.path.join(CARPETA, 'scaler.joblib'))
    encoder = load(os.path.join(CARPETA, 'encoder.joblib'))
    label_encoder = load(os.path.join(CARPETA, 'label_encoder.joblib'))
    model = load(os.path.join(CARPETA, 'model.joblib'))
    return scaler, encoder, label_encoder, model

try:
    scaler, encoder, label_encoder, model = cargar_recursos()
    recursos_cargados = True
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")
    st.info("Asegúrate de que la carpeta 'resultados_u4a1' contenga los archivos .joblib generados en el notebook.")
    recursos_cargados = False

# Título y descripción
st.title("🚗 Clasificación de Gravedad de Accidentes (RUNT)")
st.markdown("""
Esta aplicación predice si el accidente en el que se involucra un vehículo resulta 
**CON HERIDOS** o **CON MUERTOS**, basándose en las características del vehículo y el evento.
""")

if recursos_cargados:
    st.header("Formulario de entrada de datos")
    
    # Crear columnas para el diseño del formulario
    col1, col2 = st.columns(2)
    
    with col1:
        marca = st.selectbox(
            "Marca del Vehículo",
            options=["AKT", "YAMAHA", "CHEVROLET", "SUZUKI", "HONDA", "BAJAJ", "TOYOTA", "MAZDA", "FORD", "RENAULT", "OTRO"]
        )
        tipo = st.selectbox(
            "Tipo de Vehículo",
            options=["MOTOCICLETA", "AUTOMOVIL", "CAMIONETA", "CAMION", "TRACTOCAMION", "BUS", "MICROBUS", "OTRO"]
        )
        departamento = st.selectbox(
            "Departamento del Accidente",
            options=["ANTIOQUIA", "BOGOTA D.C.", "VALLE DEL CAUCA", "SANTANDER", "CUNDINAMARCA", "ATLANTICO", "BOLIVAR", "OTRO"]
        )
        municipio = st.text_input("Municipio del Accidente", value="MEDELLIN")
        
    with col2:
        anio_accidente = st.number_input(
            "Año del Accidente",
            min_value=2015,
            max_value=2026,
            value=2024,
            step=1
        )
        mes_accidente = st.slider(
            "Mes del Accidente",
            min_value=1,
            max_value=12,
            value=6
        )
        antiguedad = st.number_input(
            "Antigüedad del Vehículo (Años)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    st.markdown("---")
    
    if st.button("🎯 Realizar Predicción", use_container_width=True):
        # Crear un DataFrame con las entradas para que coincida con el preprocesamiento
        entrada_df = pd.DataFrame([{
            'MARCA_VEHICULO': marca.upper(),
            'TIPO_VEHICULO': tipo.upper(),
            'DEPARTAMENTO_ACCIDENTE': departamento.upper(),
            'MUNICIPIO_ACCIDENTE': municipio.upper(),
            'ANIO_ACCIDENTE': float(anio_accidente),
            'MES_ACCIDENTE': float(mes_accidente),
            'ANTIGUEDAD_EVENTO': float(antiguedad)
        }])
        
        # Aplicar exactamente los mismos pasos de transformación independientes
        try:
            NUMERICAS = ['ANIO_ACCIDENTE', 'MES_ACCIDENTE', 'ANTIGUEDAD_EVENTO']
            CATEGORICAS = ['MARCA_VEHICULO', 'TIPO_VEHICULO', 'DEPARTAMENTO_ACCIDENTE', 'MUNICIPIO_ACCIDENTE']
            
            # Escalado numérico e imputación/codificación categórica utilizando los transformadores guardados
            numeros = scaler.transform(entrada_df[NUMERICAS])
            categorias = encoder.transform(entrada_df[CATEGORICAS])
            
            # Construir la matriz dispersa de entrada
            matriz_entrada = hstack([csr_matrix(numeros), csr_matrix(categorias)], format='csr')
            
            # Predicción
            prediccion_numerica = model.predict(matriz_entrada).astype(int)[0]
            resultado_final = label_encoder.inverse_transform([prediccion_numerica])[0]
            
            # Obtener probabilidades si el modelo lo permite
            if hasattr(model, "predict_proba"):
                probabilidades = model.predict_proba(matriz_entrada)[0]
                prob_porcentaje = probabilidades[prediccion_numerica] * 100
                detalle_prob = f"con un nivel de confianza del {prob_porcentaje:.2f}%"
            else:
                detalle_prob = ""
                
            # Mostrar el resultado al usuario con estilo visual diferenciado
            st.subheader("Resultado de la Predicción:")
            if resultado_final == "CON MUERTOS":
                st.error(f"🚨 Predicción: {resultado_final} {detalle_prob}")
            else:
                st.success(f"✅ Predicción: {resultado_final} {detalle_prob}")
                
        except Exception as err:
            st.error(f"Ocurrió un error al procesar la predicción: {err}")
