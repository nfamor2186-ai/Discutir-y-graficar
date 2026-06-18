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
# CONFIGURACIÓN DE LA API DE GEMINI (NUEVA LIBRERÍA OFICIAL GOOGLE-GENAI)
# =========================================================================
# Recupera la clave desde los Secrets de Streamlit o variables de entorno locales
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ No se detectó la clave `GEMINI_API_KEY`. Por favor, agrégala en los Secrets de Streamlit o variables de entorno.")
else:
    # Inicialización del cliente oficial de Google GenAI
    client = genai.Client(api_key=api_key)

# =========================================================================
# BARRA LATERAL: ENTRADA DE LA RELACIÓN
# =========================================================================
st.sidebar.header("📝 Entrada de la Ecuación")
st.sidebar.markdown("Ingresa la ecuación de la relación en el plano:")

# Ejemplo estándar basado en tus ejercicios anteriores
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
    if not api_key:
        st.error("No se puede calcular sin una API Key válida configurada.")
    else:
        # 1. Crear el prompt adaptado al estilo didáctico de tu archivo .tex
        prompt_matematico = f"""
        Actúa como un profesor experto en Matemática Básica y Álgebra Universitaria. 
        Debes resolver de manera completamente analítica, rigurosa y paso a paso la discusión de la siguiente relación geométrica: 
        
        {ecuacion_str} = 0 (o adaptada a su forma implícita equivalente).
        
        Es absolutamente obligatorio que sigues de forma estricta los siguientes 5 pasos del método clásico de discusión de curvas, estructurando las ecuaciones en formato Markdown con LaTeX (utilizando $$ para bloques y $ para ecuaciones en línea):
        
        1) EXTENSIÓN: 
           a) Dominio: Despeja 'y' en función de 'x' (si es posible), analiza analíticamente las restricciones en el campo real (ej. denominadores distintos de cero, raíces no negativas) y define el conjunto del dominio de forma matemática clara.
           b) Rango: Utiliza el método algebraico clásico. Si es una función racional con términos cuadráticos, reordénala como una ecuación de segundo grado de la forma Ax^2 + Bx + C = 0 y aplica el criterio del discriminante (Delta >= 0) para hallar los intervalos válidos de 'y'. Muestra este proceso detalladamente.
        
        2) INTERSECCIONES CON LOS EJES:
           a) Con el Eje X (Haciendo y = 0). Muestra la ecuación resultante y sus soluciones coordenades (x, 0).
           b) Con el Eje Y (Haciendo x = 0). Muestra la ecuación resultante y sus soluciones coordenades (0, y).
        
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
                # Llamada al modelo estable recomendado para razonamiento de texto y código
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_matematico,
                )
                
                # Renderizar la respuesta matemática en formato markdown nativo
                st.header("Análisis Completo de la Relación (Resolución por IA)")
                st.markdown(response.text)
                st.divider()
                
            except Exception as e:
                st.error(f"Error al conectar con el motor de Gemini: {e}")

        # =========================================================================
        # 4. RENDERIZADO GRÁFICO SEGURO (USANDO MATPLOTLIB NUMÉRICO)
        # =========================================================================
        st.subheader("📈 Representación Geométrica de Control")
        st.markdown("Gráfica generada numéricamente para validar el comportamiento analítico descrito arriba:")

        try:
            # Limpieza básica para interpretar la función en numpy de forma segura
            # Si el usuario ingresa algo como "y = (2*x-3)/(x**2+x-2)", aislamos el miembro derecho
            expr_numpy = ecuacion_str
            if "=" in expr_numpy:
                partes = expr_numpy.split("=")
                if partes[0].strip() == "y":
                    expr_numpy = partes[1].strip()
                elif partes[1].strip() == "y":
                    expr_numpy = partes[0].strip()

            # Reemplazar operadores a formato compatible con python si el usuario usó notación común
            expr_numpy = expr_numpy.replace("^", "**")

            # Crear malla de puntos descartando valores extremos
            x_vals = np.linspace(-7, 7, 600)
            
            # Evaluar la cadena de manera segura usando eval sobre el espacio local de numpy
            # Para evitar que divisiones por cero rompan el código, usamos un diccionario de funciones matemáticas
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

            # Configuración estética de la gráfica estilo pgfplots/LaTeX
            fig, ax = plt.subplots(figsize=(9, 6.5))
            
            # Enmascarar saltos asintóticos verticales bruscos para que la gráfica no dibuje líneas verticales erróneas
            posibles_saltos = np.where(np.abs(np.diff(y_vals)) > 10)[0]
            if len(posibles_saltos) > 0:
                # Si hay discontinuidades, graficamos por tramos continuos
                tramos = np.split(x_vals, posibles_saltos + 1)
                tramos_y = np.split(y_vals, posibles_saltos + 1)
                primera_etiqueta = True
                for tx, ty in zip(tramos, tramos_y):
                    # Filtrar valores que se vayan demasiado al infinito para la visualización fija
                    filtro = (ty > -15) & (ty < 15)
                    if primera_etiqueta:
                        ax.plot(tx[filtro], ty[filtro], color='blue', linewidth=2, label=f"Relación: {ecuacion_str}")
                        primera_etiqueta = False
                    else:
                        ax.plot(tx[filtro], ty[filtro], color='blue', linewidth=2)
            else:
                ax.plot(x_vals, y_vals, color='blue', linewidth=2, label=f"Relación: {ecuacion_str}")

            # Ejes cartesianos principales
            ax.axhline(0, color='black', linewidth=1.2)
            ax.axvline(0, color='black', linewidth=1.2)
            
            # Grilla fina discontinua
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Límites de visualización estándar para cuadernos de ejercicios
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
            ax.set_xlabel('$x$', fontsize=11, loc='right')
            ax.set_ylabel('$y$', fontsize=11, loc='top')
            ax.legend(loc='upper right', fontsize='small')
            
            st.pyplot(fig)

        except Exception as e_plot:
            st.info("Nota sobre el gráfico: La relación ingresada puede requerir una gráfica implícita compleja o contiene variables no despejadas de forma explícita directa. Apóyate en los puntos de tabulación provistos por el asistente de arriba.")