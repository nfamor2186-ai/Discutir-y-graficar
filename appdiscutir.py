import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from google import genai
import os

# Configuración de la interfaz web
st.set_page_config(page_title="Discusión de Relaciones con IA", page_icon="📐", layout="wide")

st.title("📐 Laboratorio Didáctico: Discusión y Gráfica de Relaciones (Potenciado con IA)")
st.markdown("Analiza curvas paso a paso utilizando el método clásico de discusión de relaciones mediante inteligencia artificial.")

# =========================================================================
# CONFIGURACIÓN DE LA API DE GEMINI (AJUSTE DE SEGURIDAD DIRECTO)
# =========================================================================
# 1. Intentamos jalar la clave desde los Secrets de Streamlit o del entorno
api_key_env = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

client = None
if api_key_env:
    try:
        # Pasamos la clave directamente al inicializador para asegurar que no falle
        client = genai.Client(api_key=api_key_env)
    except Exception as e_init:
        st.error(f"Error al inicializar el cliente de IA: {e_init}")
else:
    st.error("⚠️ No se detectó la clave `GEMINI_API_KEY`. Por favor, agrégala en los Secrets de Streamlit Cloud o en tu entorno local.")

# =========================================================================
# BARRA LATERAL: ENTRADA DE LA RELACIÓN
# =========================================================================
st.sidebar.header("📝 Entrada de la Ecuación")
st.sidebar.markdown("Ingresa la ecuación de la relación en el plano:")

eq_def = "y = (2*x - 3)/(x**2 + x - 2)"
ecuacion_str = st.sidebar.text_input("Expresión de la curva:", value=eq_def)

st.sidebar.info("💡 **Ejemplos que puedes probar:**\n"
                "• `y = (2*x - 3)/(x**2 + x - 2)`\n"
                "• `x**2 - y**2 = 4`\n"
                "• `x**2 + y - 2*x - 3 = 0`")

calcular = st.sidebar.button("Discutir y Graficar Relación", type="primary")

# =========================================================================
# FLUJO PRINCIPAL DE PROCESAMIENTO
# =========================================================================
if calcular:
    if client is None:
        st.error("No se puede realizar el cálculo porque la API Key no es válida o no está configurada en Streamlit.")
    else:
        # Prompt pedagógico adaptado al estilo estructurado de tu pizarra
        prompt_matematico = f"""
        Actúa como un profesor experto en Matemática Básica y Álgebra Universitaria. 
        Debes resolver de manera completamente analítica, rigurosa y paso a paso la discusión de la siguiente relación geométrica: 
        
        {ecuacion_str} = 0 (o adaptada a su forma implícita equivalente).
        
        Es absolutamente obligatorio que sigas de forma estricta los siguientes 5 pasos del método clásico de discusión de curvas, estructurando las ecuaciones en formato Markdown con LaTeX (utilizando $$ para bloques y $ para ecuaciones en línea):
        
        1) EXTENSIÓN: 
           a) Dominio: Despeja 'y' en función de 'x' (si es posible), analiza analíticamente las restricciones en el campo real (ej. denominadores distintos de cero, raíces no negativas) y define el conjunto del dominio de forma matemática clara.
           b) Rango: Utiliza el método algebraico clásico. Si es una función racional con términos cuadráticos, reordénala como una ecuación de segundo grado de la forma Ax^2 + Bx + C = 0 y aplica el criterio del discriminante (Delta >= 0) para hallar los intervalos válidos de 'y'. Muestra este proceso detalladamente.
        
        2) INTERSECCIONES CON LOS EJES:
           a) Con el Eje X (Haciendo y = 0). Muestra la ecuación resultante y sus soluciones coordenadas (x, 0).
           b) Con el Eje Y (Haciendo x = 0). Muestra la ecuación resultante y sus soluciones coordenadas (0, y).
        
        3) SIMETRÍAS: Analiza algebraicamente sustituyendo las variables y evalúa la paridad formal:
           - Con respecto al Eje X (evaluando si E(x, y) = E(x, -y)).
           - Con respecto al Eje Y (evaluando si E(x, y) = E(-x, y)).
           - Con respecto al Origen (evaluando si E(x, y) = E(-x, -y)).
        
        4) ASÍNTOTAS ALGEBRAICAS:
           - Verticales: Halla los valores de 'x' que anulan el denominador de la relación.
           - Horizontales: Analiza los valores de 'y' prohibitivos o aplica la comparación algebraica de grados de los polinomios.
        
        5) TABLA DE VALORES SUCINTA: Proporciona una pequeña lista de puntos clave tabulados (al menos 4 puntos en formato de coordenadas claras) para orientar el dibujo de los estudiantes.
        
        Evita usar tecnicismos de cálculo diferencial (nada de límites, ni derivadas, ni integrales). Todo debe ser álgebra pura de nivel preuniversitario o de matemática básica elemental.
        """

        with st.spinner("Gemini analizando y resolviendo la relación paso a paso..."):
            try:
                # Generación usando el modelo estable recomendado
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_matematico,
                )
                
                st.header("Análisis Completo de la Relación (Resolución por IA)")
                st.markdown(response.text)
                st.divider()
                
            except Exception as e_api:
                st.error(f"Error al generar el contenido con Gemini: {e_api}. Por favor verifica que tu clave copiada en Secrets sea la correcta.")

        # =========================================================================
        # RENDERIZADO GRÁFICO SEGURO (USANDO MATPLOTLIB NUMÉRICO)
        # =========================================================================
        st.subheader("📈 Representación Geométrica de Control")
        st.markdown("Gráfica generada numéricamente para validar el comportamiento analítico descrito arriba:")

        try:
            expr_numpy = ecuacion_str
            if "=" in expr_numpy:
                partes = expr_numpy.split("=")
                if partes[0].strip() == "y":
                    expr_numpy = partes[1].strip()
                elif partes[1].strip() == "y":
                    expr_numpy = partes[0].strip()

            expr_numpy = expr_numpy.replace("^", "**")
            x_vals = np.linspace(-7, 7, 600)
            
            dict_local = {
                "x": x_vals,
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "sqrt": np.sqrt,
                "exp": np.exp
            }
            
            with np.errstate(divide='ignore', invalid='ignore'):
                y_vals = eval(expr_numpy, {"__builtins__": None}, dict_local)

            fig, ax = plt.subplots(figsize=(9, 6.5))
            
            posibles_saltos = np.where(np.abs(np.diff(y_vals)) > 10)[0]
            if len(posibles_saltos) > 0:
                tramos = np.split(x_vals, posibles_saltos + 1)
                tramos_y = np.split(y_vals, posibles_saltos + 1)
                primera_etiqueta = True
                for tx, ty in zip(tramos, tramos_y):
                    filtro = (ty > -15) & (ty < 15)
                    if primera_etiqueta:
                        ax.plot(tx[filtro], ty[filtro], color='blue', linewidth=2, label=f"Curva: {ecuacion_str}")
                        primera_etiqueta = False
                    else:
                        ax.plot(tx[filtro], ty[filtro], color='blue', linewidth=2)
            else:
                ax.plot(x_vals, y_vals, color='blue', linewidth=2, label=f"Curva: {ecuacion_str}")

            ax.axhline(0, color='black', linewidth=1.2)
            ax.axvline(0, color='black', linewidth=1.2)
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
            ax.set_xlabel('$x$', fontsize=11, loc='right')
            ax.set_ylabel('$y$', fontsize=11, loc='top')
            ax.legend(loc='upper right', fontsize='small')
            
            st.pyplot(fig)

        except Exception as e_plot:
            st.info("Nota sobre la gráfica: Expresión implícita detectada.")
