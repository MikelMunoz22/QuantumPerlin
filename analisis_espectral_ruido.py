# ==============================================================================
# ANÁLISIS ESPECTRAL DE RUIDO BLANCO, PERLIN Y BROWNIANO FRACTAL (1D Y 2D)
# ==============================================================================
# Junta en un solo script lo que hacían generacion_reuido_perlinV4comparacion.py
# (generación 1D/2D, FFT, coeficientes de mayor peso y su distribución sobre
# MUCHAS muestras) y bueno.py (frecuencia en nº de ciclos, fase, ajuste de la
# caída del espectro con alfa, beta y dB/octava, y dispersión de fases).
# Con la misma semilla y los mismos parámetros da exactamente las mismas
# muestras que esos dos scripts.

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d


# ==============================================================================
# 0. CONFIGURACIÓN USUARIO
# ==============================================================================

CONFIG = {
    # Opción de ruido: "perlin", "blanco" o "browniano"
    # Opción de dimensión del análisis: "1D" o "2D"
    # Se ejecutan TODAS las combinaciones de las dos listas (p.ej. 3 tipos de
    # ruido x 2 dimensiones = 6 combinaciones).
    "TIPOS_RUIDO": ["perlin"],
    "DIMENSIONES": ["1D"],

    # Parámetros globales
    "NUM_BITS": 11,                 # tamaño = 2**num_bits puntos en el eje X (n=5 -> 32 puntos)
    "RESOLUCION": 64,               # nº de celdas de la rejilla Perlin por eje
    "NUM_MUESTRAS": 1,              # muestras sueltas a dibujar (0 para saltárselo)
    "COEFS_RELEVANTES": 30,         # nº de coeficientes de mayor peso (top-N)

    # Parámetros del ruido Browniano fractal (solo se usan si el tipo es "browniano")
    "OCTAVAS": 7,
    "PERSISTENCIA": 0.707,
    "LACUNARIDAD": 2,

    # Parámetros del análisis de distribución de coeficientes: genera MUCHAS
    # muestras de un mismo tipo de ruido y agrega sus coeficientes de mayor
    # peso, para ver si caen sistemáticamente en las mismas frecuencias en vez
    # de mirar solo "NUM_MUESTRAS" muestras sueltas.
    "ANALIZAR_DISTRIBUCION": False,
    "NUM_MUESTRAS_ANALISIS": 100,

    # Frecuencia en "ciclos" (k = nº de ciclos completos en la señal, como en
    # bueno.py) o "normalizada" (k/N, como en la V4)
    "UNIDAD_FRECUENCIA": "ciclos",
    "EXPONENTE_REFERENCIA": 2,      # curva de referencia 1/k^p en la gráfica de potencia
    "DIBUJAR": True,                # si es False, ni siquiera se generan las figuras
    "GUARDAR_FIGURAS": False,       # si es True, se guardan como PNG en CARPETA_FIGURAS
    "CARPETA_FIGURAS": "figuras",
}

TIPOS_VALIDOS = ("blanco", "perlin", "browniano")
DIMENSIONES_VALIDAS = ("1D", "2D")
NOMBRE_RUIDO = {
    "blanco": "Ruido Blanco",
    "perlin": "Ruido Perlin",
    "browniano": "Ruido Browniano Fractal",
}
MARCAS_PI = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
ETIQUETAS_PI = ["0", "π/2", "π", "3π/2", "2π"]


# ==============================================================================
# 1. GENERACIÓN DE RUIDO
# ==============================================================================
# En 1D la señal se trata como variable discreta: el número de bits "n" fija
# directamente cuántos puntos tiene el eje X, tamaño = 2**n (p.ej. n=5 -> 32
# puntos). No hay cuantización de los VALORES del ruido (siguen siendo números
# reales); lo discreto es el dominio X, y por eso se dibuja con escalones en vez
# de una línea continua (ver figura_muestra_1d).

def multiplo_requerido(tipo, cfg):
    # perlin_numpy exige que el tamaño sea múltiplo de la resolución; en el
    # Browniano, de la resolución de la octava más fina:
    # "resolución * lacunaridad^(octavas-1)".
    if tipo == "perlin":
        return cfg["RESOLUCION"]
    if tipo == "browniano":
        return cfg["RESOLUCION"] * cfg["LACUNARIDAD"] ** (cfg["OCTAVAS"] - 1)
    return 1


def generar_ruido(semilla, tipo, dimension, cfg):
    # La semilla se aplica aquí, justo antes de consumir números aleatorios
    # (perlin_numpy usa el generador global de NumPy).
    np.random.seed(seed=semilla)
    tamaño = 2 ** cfg["NUM_BITS"]
    res = cfg["RESOLUCION"]

    if tipo == "blanco":
        ruido_1d = np.random.uniform(-0.5, 0.5, size=tamaño)
        if dimension == "1D":
            return ruido_1d
        # El campo 2D se sortea DESPUÉS de la fila 1D, igual que en la V4, para
        # que la misma semilla siga dando la misma muestra.
        return np.random.uniform(-0.5, 0.5, size=(tamaño, tamaño))

    # Optimización en modo "1D": solo se usa la fila 0 de la textura, así que no
    # hace falta generarla completa (tamaño x tamaño). perlin_numpy sortea los
    # ángulos de gradiente en función únicamente de la resolución (res), no del
    # tamaño de salida (shape); y la posición fraccional de la fila 0 dentro de
    # su celda es siempre 0 (en TODAS las octavas), sea cual sea el número de
    # filas pedido. Por tanto, basta con pedir el mínimo de filas que exige la
    # divisibilidad y da exactamente el mismo resultado en la fila 0, evitando
    # calcular valores que nunca se llegaban a usar.
    # En modo "2D" hace falta el campo completo, así que se piden las "tamaño"
    # filas enteras.
    filas = tamaño if dimension == "2D" else multiplo_requerido(tipo, cfg)

    if tipo == "perlin":
        ruido_2d = generate_perlin_noise_2d((filas, tamaño), (res, res))
    else:
        # Ruido Browniano fractal (fBm): suma de varias octavas de ruido Perlin,
        # cada una al doble (lacunaridad) de frecuencia y a la mitad
        # (persistencia) de amplitud que la anterior.
        ruido_2d = generate_fractal_noise_2d(
            (filas, tamaño),
            (res, res),
            octaves=cfg["OCTAVAS"],
            persistence=cfg["PERSISTENCIA"],
            lacunarity=cfg["LACUNARIDAD"],
        )
    return ruido_2d if dimension == "2D" else ruido_2d[0, :]


# ==============================================================================
# 2. CÁLCULO DE FFT, COEFICIENTES DE MAYOR PESO Y AJUSTE DE LA CAÍDA
# ==============================================================================

def calcular_espectro(ruido):
    # Vale para 1D y 2D: rfftn aprovecha que la señal es real y solo transforma
    # la mitad+1 de la última dimensión; cada eje necesita su propio vector de
    # frecuencias.
    X = np.fft.rfftn(ruido)

    # La frecuencia NO se normaliza (k/N): se quiere el valor real, el número de
    # ciclos completos que esa componente completa a lo largo de toda la señal,
    # es decir k directamente (0, 1, 2, ..., N/2). En el último eje el índice
    # del array de coeficientes ES k, así que basta con np.arange. En 2D el
    # primer eje viene de fftfreq(), que da primero 0 y las positivas y luego
    # las negativas.
    frecuencias = [np.rint(np.fft.fftfreq(m) * m) for m in ruido.shape[:-1]]
    frecuencias.append(np.arange(X.shape[-1], dtype=float))
    return X, frecuencias


def modulo_k(frecuencias):
    # |k| de cada coeficiente (en 1D es simplemente k)
    if len(frecuencias) == 1:
        return np.abs(frecuencias[0])
    return np.hypot(frecuencias[0][:, None], frecuencias[1][None, :])


def coefs_mayor_peso(X, N):
    # Obtener los N índices mayores. Vale para 1D y 2D: se aplana la matriz de
    # magnitudes para sacar el top-N con argpartition, y con unravel_index se
    # recuperan la fila/columna de cada coeficiente para poder mapearlas a su
    # (frecuencia_x, frecuencia_y). En 1D el aplanado no cambia nada.
    mags = np.abs(X).ravel()
    idx_top = np.argpartition(mags, -N)[-N:]
    idx_ordenados = idx_top[np.argsort(-mags[idx_top])]
    return np.unravel_index(idx_ordenados, X.shape)


def ajustar_caida(X, frecuencias):
    # Ajuste de ley de potencias de la caída del espectro, P(k) ∝ 1/k^beta, sin
    # k=0. Como la potencia es |c_k|², ajustar la magnitud da exactamente la
    # mitad de pendiente, así que alfa = beta/2 sin hacer un segundo ajuste.
    potencia = np.abs(X) ** 2

    if X.ndim == 1:
        k, P = frecuencias[0][1:], potencia[1:]
    else:
        # En 2D se usa la media radial de la potencia: se agrupan los
        # coeficientes por |k| entero, hasta Nyquist.
        radio = np.rint(modulo_k(frecuencias)).astype(int).ravel()
        suma = np.bincount(radio, weights=potencia.ravel())
        cuenta = np.bincount(radio)
        k_max = min(len(frecuencias[0]), 2 * (len(frecuencias[1]) - 1)) // 2
        k = np.arange(1, k_max + 1, dtype=float)
        P = suma[1:k_max + 1] / cuenta[1:k_max + 1]

    validos = P > 0  # el logaritmo de 0 no existe
    k, P = k[validos], P[validos]
    pendiente_beta, intercepto_beta = np.polyfit(np.log(k), np.log(P), 1)
    beta = -pendiente_beta

    # Equivalente de la pendiente en dB/octava: una octava es doblar la
    # frecuencia, así que dB/octava = 20*pendiente_alfa*log10(2)
    # = 10*pendiente_beta*log10(2)
    return {
        "k": k,
        "potencia": P,
        "potencia_ajustada": np.exp(intercepto_beta) * k ** pendiente_beta,
        "beta": beta,
        "alfa": beta / 2,
        "db_octava": 10 * pendiente_beta * np.log10(2),
    }


# ==============================================================================
# 3. FUNCIONES DE VISUALIZACIÓN
# ==============================================================================

def escala_frecuencia(cfg):
    return 1.0 if cfg["UNIDAD_FRECUENCIA"] == "ciclos" else 1.0 / 2 ** cfg["NUM_BITS"]


def etiqueta_frecuencia(cfg, eje="", modulo=False):
    simbolo = "|k|" if modulo else f"k{eje}"
    if cfg["UNIDAD_FRECUENCIA"] == "ciclos":
        return f"{simbolo} (nº de ciclos)"
    return f"{simbolo}/N (frecuencia normalizada)"


def barra_de_fase(fig, ax):
    sm = plt.cm.ScalarMappable(cmap="twilight", norm=Normalize(-np.pi, np.pi))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cbar.set_label("Fase (rad)")


def figura_muestra_1d(ruido, X, frecuencias, top, titulo, cfg):
    # Figura 1: señal y espectro (con los top resaltados encima)
    esc = escala_frecuencia(cfg)
    frecuencia, magnitud = frecuencias[0] * esc, np.abs(X)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"{titulo} ({len(ruido)} puntos)", fontsize=13, fontweight="bold")

    # Panel 1: señal como variable discreta (escalones rellenos: se ve igual
    # que con barras pero se dibuja mucho más rápido cuando hay miles de puntos)
    axes[0].stairs(ruido, np.arange(len(ruido) + 1) - 0.5, baseline=0, fill=True, color="tab:blue")
    axes[0].set_title("Señal (variable discreta)")
    axes[0].set_xlabel("Posición X (entero)")
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: espectro completo (escala log en Y, para que el coeficiente k=0
    # no aplaste visualmente al resto) con los top-N resaltados encima en vez de
    # repetirlos en un panel aparte.
    axes[1].stem(frecuencia, magnitud, linefmt="tab:blue", markerfmt=" ", basefmt=" ")
    axes[1].stem(frecuencia[top[0]], magnitud[top], linefmt="red", markerfmt="rD", basefmt=" ")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Espectro completo ({len(magnitud)} coeficientes) — top {len(top[0])} en rojo")
    axes[1].set_xlabel(etiqueta_frecuencia(cfg))
    axes[1].set_ylabel("Magnitud |c_k| (log)")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def figura_muestra_2d(ruido, X, frecuencias, top, titulo, cfg):
    # Versión 2D de figura_muestra_1d(): mapa de calor del campo de ruido, mapa
    # de calor del espectro FFT 2D completo, y dispersión de los N coeficientes
    # de mayor peso sobre el plano (frecuencia_x, frecuencia_y).
    esc = escala_frecuencia(cfg)
    coef_fourier = np.abs(X)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"{titulo} ({ruido.shape[0]}×{ruido.shape[1]} puntos)", fontsize=13, fontweight="bold")

    # --- Subgráfico 1: Campo de Ruido ---
    mapa = axes[0].imshow(ruido, cmap="viridis", origin="lower")
    axes[0].set_title("Mapa de ruido")
    axes[0].set_xlabel("Posición X")
    axes[0].set_ylabel("Posición Y")
    fig.colorbar(mapa, ax=axes[0], fraction=0.046)

    # --- Subgráfico 2: Espectro FFT 2D Completo ---
    # frecuencias_x viene de fftfreq(), que NO está ordenada de menor a mayor
    # (da primero 0 y las positivas, luego las negativas). Para poder usar
    # "extent" y que la posición en la imagen se corresponda linealmente con la
    # etiqueta de frecuencia, hay que reordenar esa fila con fftshift (tanto los
    # datos como el propio vector de frecuencias). frecuencias_y viene de
    # rfftfreq(), que ya está ordenada (0 a Nyquist), así que no necesita
    # reordenarse.
    #
    # coef_fourier tiene forma (filas=frecuencia_x, columnas=frecuencia_y).
    # imshow pone el eje 0 del array en vertical y el eje 1 en horizontal, así
    # que sin más lo mostraría con X en vertical e Y en horizontal — al revés
    # que los paneles 1 y 3 (X horizontal, Y vertical). Trasponemos para que los
    # tres paneles compartan la misma orientación.
    #
    # El color va en escala log, por lo mismo que en 1D: para que k=0 no aplaste
    # visualmente al resto.
    frecuencias_x_centrado = np.fft.fftshift(frecuencias[0]) * esc
    frecuencias_y = frecuencias[1] * esc
    espectro_para_dibujar = np.fft.fftshift(coef_fourier, axes=0).T
    medio = esc / 2  # cada píxel se centra en su frecuencia
    extent = [
        frecuencias_x_centrado[0] - medio, frecuencias_x_centrado[-1] + medio,
        frecuencias_y[0] - medio, frecuencias_y[-1] + medio,
    ]
    espectro = axes[1].imshow(
        espectro_para_dibujar, cmap="magma", origin="lower", aspect="auto", extent=extent,
        norm=LogNorm(vmin=espectro_para_dibujar[espectro_para_dibujar > 0].min(),
                     vmax=espectro_para_dibujar.max()),
    )
    axes[1].set_title(f"Espectro FFT 2D ({coef_fourier.size} coeficientes, escala log)")
    axes[1].set_xlabel(etiqueta_frecuencia(cfg, "x"))
    axes[1].set_ylabel(etiqueta_frecuencia(cfg, "y"))
    fig.colorbar(espectro, ax=axes[1], fraction=0.046, label="|c_k|")

    # --- Subgráfico 3: Coeficientes de Mayor Peso ---
    dispersión = axes[2].scatter(
        frecuencias[0][top[0]] * esc, frecuencias[1][top[1]] * esc, c=coef_fourier[top],
        cmap="magma", s=90, edgecolor="k",
    )
    axes[2].set_title(f"Top {len(top[0])} coeficientes de mayor peso")
    axes[2].set_xlabel(etiqueta_frecuencia(cfg, "x"))
    axes[2].set_ylabel(etiqueta_frecuencia(cfg, "y"))
    axes[2].grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(dispersión, ax=axes[2], fraction=0.046, label="|c_k|")

    fig.tight_layout()
    return fig


def figura_caida(ajuste, titulo, radial, cfg):
    # Figura 2: ajuste de ley de potencias, en log-log y en dB/octava. En 2D
    # se dibuja la media radial de la potencia (ver ajustar_caida).
    esc = escala_frecuencia(cfg)
    k = ajuste["k"]
    frecs = k * esc
    potencia, potencia_ajustada = ajuste["potencia"], ajuste["potencia_ajustada"]
    p = cfg["EXPONENTE_REFERENCIA"]
    eje_x = etiqueta_frecuencia(cfg, modulo=radial) + ", log"

    fig, (ax_loglog, ax_beta, ax_db) = plt.subplots(1, 3, figsize=(18, 5))
    sufijo = " (media radial)" if radial else ""
    fig.suptitle(f"{titulo}: ajuste de la caída del espectro{sufijo}", fontsize=13, fontweight="bold")

    # Panel izquierdo: magnitud (alfa)
    ax_loglog.loglog(frecs, np.sqrt(potencia), color="tab:blue", linewidth=0.8, label="Espectro")
    ax_loglog.loglog(frecs, np.sqrt(potencia_ajustada), color="red", linewidth=2,
                     label=f"Ajuste (α = {ajuste['alfa']:.2f})")
    ax_loglog.set_title("Escala log-log — magnitud")
    ax_loglog.set_xlabel(eje_x)
    ax_loglog.set_ylabel("Magnitud |c_k| (log)")

    # Panel central: potencia (beta) — mismo patrón, con potencia en vez de
    # magnitud. Curva 1/k^p continua, anclada al primer coeficiente.
    ax_beta.loglog(frecs, potencia, color="tab:blue", linewidth=0.8, label="Espectro de potencia")
    ax_beta.loglog(frecs, potencia_ajustada, color="red", linewidth=2,
                   label=f"Ajuste (β = {ajuste['beta']:.2f})")
    ax_beta.loglog(frecs, potencia[0] * (k / k[0]) ** (-p), color="green", linewidth=2,
                   label=f"$1/k^{{{p}}}$")
    ax_beta.set_title("Escala log-log — potencia")
    ax_beta.set_xlabel(eje_x)
    ax_beta.set_ylabel("Potencia |c_k|² (log)")

    for ax in (ax_loglog, ax_beta):
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    # Panel derecho: conversión a decibelios, dB = 20*log10(magnitud), que es
    # lo mismo que 10*log10(potencia)
    ampli_dB = 10 * np.log10(potencia)
    ax_db.fill_between(frecs, ampli_dB, ampli_dB.min() - 5, step="mid", color="tab:blue", alpha=0.85)
    ax_db.plot(frecs, 10 * np.log10(potencia_ajustada), color="red", linewidth=2)
    ax_db.text(0.62, 0.88, f"{ajuste['db_octava']:.1f} dB/Oct", transform=ax_db.transAxes,
               color="red", fontsize=16, fontweight="bold")
    ax_db.set_xscale("log")
    ax_db.set_title("Escala dB vs. frecuencia")
    ax_db.set_xlabel(eje_x)
    ax_db.set_ylabel("Magnitud (dB)")
    ax_db.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def figura_fases(X, frecuencias, top, titulo, cfg):
    # Figura 3: los coeficientes top coloreados por fase, y la dispersión de
    # fases (0 a 2π) de todos los coeficientes frente a k.
    esc = escala_frecuencia(cfg)
    magnitud, fase, k_mod = np.abs(X), np.angle(X), modulo_k(frecuencias)
    mask = k_mod[top] > 0  # excluir k=0: domina la escala y su fase es siempre 0

    fig, (ax_top, ax_todas) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f"{titulo}: fases", fontsize=13, fontweight="bold")

    # --- Solo los coeficientes top, coloreados por fase ---
    if X.ndim == 1:
        frecs_top = frecuencias[0][top[0]][mask]
        orden = np.argsort(frecs_top)
        colores = plt.cm.twilight((fase[top][mask][orden] + np.pi) / (2 * np.pi))
        ax_top.bar(range(len(orden)), magnitud[top][mask][orden], color=colores)
        ax_top.set_xticks(range(len(orden)))
        ax_top.set_xticklabels([f"{f:.3g}" for f in frecs_top[orden] * esc],
                               rotation=45 if esc == 1 else 90)
        ax_top.set_yscale("log")
        ax_top.set_xlabel(etiqueta_frecuencia(cfg))
        ax_top.set_ylabel("Magnitud |c_k| (log)")
        ax_top.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    else:
        ax_top.scatter(
            frecuencias[0][top[0]][mask] * esc, frecuencias[1][top[1]][mask] * esc, c=fase[top][mask],
            cmap="twilight", vmin=-np.pi, vmax=np.pi, s=90, edgecolor="k",
        )
        ax_top.set_xlabel(etiqueta_frecuencia(cfg, "x"))
        ax_top.set_ylabel(etiqueta_frecuencia(cfg, "y"))
        ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.set_title("Top coeficientes coloreados por fase")
    barra_de_fase(fig, ax_top)

    # --- Dispersión de fase por coeficiente (sin k=0) ---
    # En 2D hay demasiados puntos para un scatter, así que se dibuja la densidad.
    mask_todas = k_mod > 0
    k_plot = k_mod[mask_todas] * esc
    fases_plot = fase[mask_todas] % (2 * np.pi)
    if X.ndim == 1:
        ax_todas.scatter(k_plot, fases_plot, alpha=0.5, s=15, color="tab:purple", edgecolor="none")
        ax_todas.set_title("Dispersión de fase por coeficiente")
    else:
        _, _, _, imagen = ax_todas.hist2d(k_plot, fases_plot, bins=[60, 36], cmap="Purples")
        fig.colorbar(imagen, ax=ax_todas, fraction=0.046, label="Nº de coeficientes")
        ax_todas.set_title("Dispersión de fase por coeficiente (densidad)")
    ax_todas.set_xlabel(etiqueta_frecuencia(cfg, modulo=X.ndim == 2))
    ax_todas.set_ylabel("Fase (rad)")
    ax_todas.set_yticks(MARCAS_PI)
    ax_todas.set_yticklabels(ETIQUETAS_PI)
    ax_todas.grid(True, ls="--", alpha=.4)

    fig.tight_layout()
    return fig


def figura_distribucion(k_top, coefs_todas, fases_todas, titulo, cfg):
    # Figura del análisis de distribución: dónde caen los coeficientes de mayor
    # peso de MUCHAS muestras, con qué magnitud y con qué fase.
    esc = escala_frecuencia(cfg)
    num_muestras_analisis = k_top.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{titulo}: {num_muestras_analisis} muestras, top {k_top.shape[1]} de cada una",
                 fontsize=13, fontweight="bold")

    if k_top.shape[2] == 1:
        frecs_todas = k_top[..., 0].ravel()

        # --- Histograma de frecuencia de aparición (una barra por cada k) ---
        bins = (np.arange(frecs_todas.max() + 2) - 0.5) * esc
        axes[0, 0].hist(frecs_todas * esc, bins=bins, color="tab:blue", alpha=0.8,
                        edgecolor="black" if len(bins) <= 120 else "none")
        axes[0, 0].set_xlabel(etiqueta_frecuencia(cfg))
        axes[0, 0].set_ylabel("Nº de veces en el top")

        # --- Todos los puntos superpuestos ---
        axes[0, 1].scatter(frecs_todas * esc, coefs_todas, alpha=0.25, s=25, color="tab:red", edgecolor="none")
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_xlabel(etiqueta_frecuencia(cfg))
        axes[0, 1].set_ylabel("Magnitud |c_k| (log)")
        k_mod = np.abs(frecs_todas)
    else:
        frecs_x_todas, frecs_y_todas = k_top[..., 0].ravel(), k_top[..., 1].ravel()

        # --- Mapa de calor de frecuencia de aparición (una celda por cada (kx, ky)) ---
        bins_x = (np.arange(frecs_x_todas.min(), frecs_x_todas.max() + 2) - 0.5) * esc
        bins_y = (np.arange(frecs_y_todas.min(), frecs_y_todas.max() + 2) - 0.5) * esc
        _, _, _, imagen = axes[0, 0].hist2d(frecs_x_todas * esc, frecs_y_todas * esc,
                                            bins=[bins_x, bins_y], cmap="magma")
        fig.colorbar(imagen, ax=axes[0, 0], fraction=0.046, label="Nº de veces en el top")
        axes[0, 0].set_xlabel(etiqueta_frecuencia(cfg, "x"))
        axes[0, 0].set_ylabel(etiqueta_frecuencia(cfg, "y"))

        # --- Todos los puntos superpuestos ---
        dispersión = axes[0, 1].scatter(
            frecs_x_todas * esc, frecs_y_todas * esc, c=coefs_todas, cmap="magma",
            alpha=0.35, s=25, edgecolor="none", norm=LogNorm(),
        )
        fig.colorbar(dispersión, ax=axes[0, 1], fraction=0.046, label="Magnitud |c_k|")
        axes[0, 1].set_xlabel(etiqueta_frecuencia(cfg, "x"))
        axes[0, 1].set_ylabel(etiqueta_frecuencia(cfg, "y"))
        k_mod = np.hypot(frecs_x_todas, frecs_y_todas)

    axes[0, 0].set_title("Frecuencia de aparición")
    axes[0, 1].set_title(f"Todos los coeficientes superpuestos ({len(coefs_todas)} puntos)")

    # --- Análisis de dispersión de la FASE (mismo espíritu que el de magnitud
    # de arriba, pero con la fase en vez del módulo). Se excluye k=0: su fase es
    # siempre 0 (componente continua), no aporta nada. Se expresa en el rango
    # [0, 2π) en vez de [-π, π).
    mask = k_mod > 0
    fases_02pi = fases_todas[mask] % (2 * np.pi)

    # Distribución de los valores de fase en sí (¿se concentran en algún
    # ángulo, o están repartidos uniformemente entre 0 y 2π?)
    axes[1, 0].hist(fases_02pi, bins=36, range=(0, 2 * np.pi), color="tab:purple",
                    edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Distribución de fases")
    axes[1, 0].set_xlabel("Fase (rad)")
    axes[1, 0].set_ylabel("Nº de coeficientes")
    axes[1, 0].set_xticks(MARCAS_PI)
    axes[1, 0].set_xticklabels(ETIQUETAS_PI)

    # Todas las fases superpuestas, en función de la frecuencia a la que
    # pertenece cada una (¿hay alguna frecuencia con fase más predecible?)
    axes[1, 1].scatter(k_mod[mask] * esc, fases_02pi, alpha=0.25, s=25, color="tab:purple", edgecolor="none")
    axes[1, 1].set_title(f"Todas las fases superpuestas ({len(fases_02pi)} puntos)")
    axes[1, 1].set_xlabel(etiqueta_frecuencia(cfg, modulo=k_top.shape[2] == 2))
    axes[1, 1].set_ylabel("Fase (rad)")
    axes[1, 1].set_yticks(MARCAS_PI)
    axes[1, 1].set_yticklabels(ETIQUETAS_PI)

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def figura_betas(betas, titulo):
    # Histograma de la beta de todas las muestras (¿es estable de una muestra a
    # otra, o depende mucho de la semilla?)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(betas, bins=30, color="tab:green", edgecolor="black", alpha=0.8)
    ax.axvline(betas.mean(), color="black", linestyle="--",
               label=f"β = {betas.mean():.3f} ± {betas.std():.3f}")
    ax.set_title(f"{titulo}: β de {len(betas)} muestras")
    ax.set_xlabel("β  (P(k) ∝ 1/k^β)")
    ax.set_ylabel("Nº de muestras")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def cerrar_figura(fig, nombre_archivo, cfg):
    # Guarda la figura si "GUARDAR_FIGURAS" es True y, si no se va a mostrar,
    # la cierra para no acumular figuras en memoria.
    if cfg["GUARDAR_FIGURAS"]:
        os.makedirs(cfg["CARPETA_FIGURAS"], exist_ok=True)
        fig.savefig(os.path.join(cfg["CARPETA_FIGURAS"], f"{nombre_archivo}.png"), dpi=150, bbox_inches="tight")
    if not cfg["DIBUJAR"]:
        plt.close(fig)


# ==============================================================================
# 4. ANÁLISIS
# ==============================================================================

def prefijo_archivo(tipo, dimension, cfg):
    # Nombre base de las figuras, con los parámetros dentro (al estilo de
    # "superposicion_frecuencia_reso8_1000shots_10coefs_1D_4oct_persi05_lacu2")
    prefijo = f"{tipo}_{dimension}_{cfg['NUM_BITS']}bits_reso{cfg['RESOLUCION']}"
    if tipo == "browniano":
        prefijo += f"_{cfg['OCTAVAS']}oct_persi{cfg['PERSISTENCIA']}_lacu{cfg['LACUNARIDAD']}"
    return prefijo


def analizar_muestras(tipo, dimension, cfg):
    # Pipeline completo (generación, FFT, extracción de coeficientes, ajuste y
    # dibujado) para UNA combinación de tipo de ruido y dimensión, muestra a
    # muestra. Usa los parámetros de CONFIG.
    esc = escala_frecuencia(cfg)
    hay_figuras = cfg["DIBUJAR"] or cfg["GUARDAR_FIGURAS"]

    for index in range(cfg["NUM_MUESTRAS"]):
        # 1. Generación y cálculo
        ruido = generar_ruido(index, tipo, dimension, cfg)
        X, frecuencias = calcular_espectro(ruido)

        # 2. Extracción de coeficientes relevantes y ajuste de la caída
        top = coefs_mayor_peso(X, cfg["COEFS_RELEVANTES"])
        ajuste = ajustar_caida(X, frecuencias)

        print(f"[tipo_ruido='{tipo}', dimension='{dimension}'] Muestra número: {index}")
        print(f"  Ampli: {np.abs(X[top])}")
        if dimension == "2D":
            print(f"  Frec X: {frecuencias[0][top[0]] * esc}")
            print(f"  Frec Y: {frecuencias[1][top[1]] * esc}")
        else:
            print(f"  Frec:  {frecuencias[0][top[0]] * esc}")
        print(f"  Fase:  {np.angle(X[top])}")
        print(f"  alfa = {ajuste['alfa']:.3f}, beta = {ajuste['beta']:.3f}, "
              f"{ajuste['db_octava']:.1f} dB/octava\n")

        # 3. Renderizado final de las gráficas.
        # Si "DIBUJAR" y "GUARDAR_FIGURAS" son False, ni siquiera se generan.
        if not hay_figuras:
            continue

        titulo = f"Muestra {index} — {NOMBRE_RUIDO[tipo]} {dimension}"
        nombre = f"{prefijo_archivo(tipo, dimension, cfg)}_muestra{index}"
        if dimension == "2D":
            fig = figura_muestra_2d(ruido, X, frecuencias, top, titulo, cfg)
        else:
            fig = figura_muestra_1d(ruido, X, frecuencias, top, titulo, cfg)
        cerrar_figura(fig, f"{nombre}_espectro", cfg)
        cerrar_figura(figura_caida(ajuste, titulo, dimension == "2D", cfg), f"{nombre}_caida", cfg)
        cerrar_figura(figura_fases(X, frecuencias, top, titulo, cfg), f"{nombre}_fases", cfg)


def analizar_distribucion(tipo, dimension, cfg):
    # Compara MUCHAS muestras de UNA sola combinación (tipo_ruido, dimension)
    # para ver si sus coeficientes de mayor peso caen sistemáticamente en la
    # misma zona del espectro.
    # De cada muestra solo se guardan los top-N y la beta, no el espectro entero:
    # en 2D y con muchos bits no cabrían todos en memoria.
    num_muestras_analisis = cfg["NUM_MUESTRAS_ANALISIS"]
    coefs_relevantes = cfg["COEFS_RELEVANTES"]
    ejes = 1 if dimension == "1D" else 2
    k_top = np.empty((num_muestras_analisis, coefs_relevantes, ejes))
    coefs = np.empty((num_muestras_analisis, coefs_relevantes))
    fases = np.empty((num_muestras_analisis, coefs_relevantes))
    betas = np.empty(num_muestras_analisis)

    for index in range(num_muestras_analisis):
        X, frecuencias = calcular_espectro(generar_ruido(index, tipo, dimension, cfg))
        top = coefs_mayor_peso(X, coefs_relevantes)
        for eje in range(ejes):
            k_top[index, :, eje] = frecuencias[eje][top[eje]]
        coefs[index] = np.abs(X[top])
        fases[index] = np.angle(X[top])
        betas[index] = ajustar_caida(X, frecuencias)["beta"]

    # coefs/fases tienen forma (num_muestras_analisis, coefs_relevantes); los
    # aplanamos para juntar en una sola nube todos los coeficientes de todas
    # las muestras.
    coefs_todas, fases_todas = coefs.ravel(), fases.ravel()

    print(
        f"[Distribución] tipo_ruido='{tipo}', dimension='{dimension}': {len(coefs_todas)} coeficientes "
        f"agregados de {num_muestras_analisis} muestras ({coefs_relevantes} por muestra)"
    )
    print(f"  beta = {betas.mean():.3f} ± {betas.std():.3f}  (alfa = beta/2 = {betas.mean() / 2:.3f})\n")

    # Igual que en analizar_muestras: si no se va a mostrar ni guardar, ni
    # siquiera se genera la figura.
    if cfg["DIBUJAR"] or cfg["GUARDAR_FIGURAS"]:
        titulo = f"{NOMBRE_RUIDO[tipo]} {dimension}"
        nombre = f"{prefijo_archivo(tipo, dimension, cfg)}_{num_muestras_analisis}muestras"
        cerrar_figura(figura_distribucion(k_top, coefs_todas, fases_todas, titulo, cfg),
                      f"{nombre}_distribucion", cfg)
        cerrar_figura(figura_betas(betas, titulo), f"{nombre}_beta", cfg)

    return k_top, coefs, fases, betas


def validar_configuracion(cfg):
    # Comprueba la configuración ANTES de empezar, para no descubrir a mitad de
    # un análisis largo que un tamaño no es compatible con perlin_numpy.
    cfg = dict(cfg)
    cfg["TIPOS_RUIDO"] = [t.lower() for t in cfg["TIPOS_RUIDO"]]
    cfg["DIMENSIONES"] = [d.upper() for d in cfg["DIMENSIONES"]]

    for t in cfg["TIPOS_RUIDO"]:
        if t not in TIPOS_VALIDOS:
            raise ValueError(f"Tipo de ruido '{t}' no válido. Opciones: {TIPOS_VALIDOS}")
    for d in cfg["DIMENSIONES"]:
        if d not in DIMENSIONES_VALIDAS:
            raise ValueError(f"Dimensión '{d}' no válida. Opciones: {DIMENSIONES_VALIDAS}")
    if cfg["UNIDAD_FRECUENCIA"] not in ("ciclos", "normalizada"):
        raise ValueError("UNIDAD_FRECUENCIA tiene que ser 'ciclos' o 'normalizada'")

    tamaño = 2 ** cfg["NUM_BITS"]
    for t in cfg["TIPOS_RUIDO"]:
        multiplo = multiplo_requerido(t, cfg)
        if tamaño % multiplo:
            formula = "RESOLUCION·LACUNARIDAD^(OCTAVAS-1)" if t == "browniano" else "RESOLUCION"
            raise ValueError(
                f"Con ruido '{t}', 2**NUM_BITS = {tamaño} tiene que ser múltiplo de {formula} = {multiplo}. "
                f"Sube NUM_BITS o baja RESOLUCION u OCTAVAS."
            )
    if "1D" in cfg["DIMENSIONES"] and cfg["COEFS_RELEVANTES"] > tamaño // 2 + 1:
        raise ValueError(f"En 1D solo hay {tamaño // 2 + 1} coeficientes; baja COEFS_RELEVANTES.")
    return cfg


# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL DEL SCRIPT
# ==============================================================================

def main(config=CONFIG):
    cfg = validar_configuracion(config)

    # Limpiamos cualquier gráfico residual en memoria
    plt.close('all')

    # Recorre todas las opciones de tipo_ruido x dimension y ejecuta el
    # pipeline completo en cada una.
    for tipo in cfg["TIPOS_RUIDO"]:
        for dimension in cfg["DIMENSIONES"]:
            print(f"\n===== Combinación: tipo_ruido='{tipo}', dimension='{dimension}' =====\n")
            if cfg["NUM_MUESTRAS"] > 0:
                analizar_muestras(tipo, dimension, cfg)
            if cfg["ANALIZAR_DISTRIBUCION"]:
                analizar_distribucion(tipo, dimension, cfg)

    if cfg["DIBUJAR"]:
        plt.show()


if __name__ == "__main__":
    main()
