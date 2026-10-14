#Este script permite generar una distribución de frecuencias siguiendo la regla de potencias 1/f^beta, con beta variable, y genera a su vez la señal (1D) o el terreno o textura (2D) asociado a este tipo de ruido fBm, todo hecho de manera cuántica

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


#DEFINICION DE PARAMETROS:
CONFIG = {
    "DIMENSION": "2D",          #"1D" o "2D"
    #nx, ny : numero de bits por eje (en 1D solo se usa nx)
    "NX": 10,
    "NY": 10,
    #beta : es la pendiente de la funcion de potencia. Sigue la ecuacion β = 2H + d, donde d es la dimensión de la señal (1 o 2), y para el ruido fBm H debe rondar 0.5
    "BETA": 3.3,
    #P0: componente adhoc de la continua para evitar 1/0
    "P0": 0.05,
    #k_corte: a partir de que |k| se atenua el espectro. None = N/2 (maximo permitido por el teorema de Nyquist), np.inf = sin corte
    "K_CORTE": None,

    #exponente del peso en el ajuste ridge: cuanto peso extra se le da en el ajuste a las ramas de mayor probabilidad.
    #la construcción del mapa la realiza con el 'PESO_OBJETIVO', el 'PESO_COMPARACION' unicamente sirve para comparar si mejora o no respecto a otro valor (None para no comparar)
    "PESO_OBJETIVO": 2.4,
    "PESO_COMPARACION": 0.5,

    #de donde salen los angulos aleatorios de la fase:
    #   "angulos_cuanticos": una semilla cuántica única e independiente por cada ángulo, de BITS_POR_ANGULO bits
    #   "semilla_cuantica": una única semilla cuántica de 32 bits, y los ángulos se aleatorizan con rng.uniform
    #   "clasica": la semilla clásica SEMILLA (SEMILLA+1, SEMILLA+2... para el resto de muestras), para poder repetir resultados
    "FUENTE_ANGULOS": "angulos_cuanticos",
    "BITS_POR_ANGULO": 32,
    "SEMILLA": 38,
    #N_semillas: indica cuantas muestras se generan por ejecución
    "N_SEMILLAS": 2,
    #normaliza el resultado restandole la componente continua
    "RESTAR_MEDIA": True,

    #que analisis se hacen
    "MEDIR_VS_TEORICA": True,   #medida estadística del circuito vs distribución teórica
    "N_SHOTS": 200_000,
    "VALIDAR_BETA_HURST": True, #beta y H calculados vs teóricos (semilla 0)
    "COMPARAR_SEMILLAS": True,  #semillas vs caso puro: R², MSE y correlaciones
    "ANALIZAR_ANGULOS": True,   #homogeneidad de los ángulos generados
    "DIBUJAR_CIRCUITO": True,

    "DIBUJAR": True,            #mostrar las figuras al final
    "GUARDAR_FIGURAS": True,    #en la carpeta figures/ junto al script
    "GUARDAR_DATOS": True,      #en la carpeta data/ junto al script
}


#crea los directorios junto al script (y no en la carpeta desde la que se ejecuta)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


#1) ESPECTRO OBJETIVO Y ORDEN GRAY

#pasa de binario (Hamming) a Gray
def calcular_indices_gray(n):
    idxs = np.arange(2**n, dtype=np.int64)
    return np.bitwise_xor(idxs, np.right_shift(idxs, 1))


#numero de bits de cada eje de la rejilla (1D: (n,), 2D: (ny, nx))
def bits_por_eje(forma):
    return [int(np.log2(N)) for N in forma]


#reordena la rejilla de probabilidades al orden Gray, eje por eje
def reordenar_probs_gray(probs_fisicas):
    out = probs_fisicas
    for eje, N in enumerate(probs_fisicas.shape):
        reordenado = np.zeros_like(out)
        seleccion = [slice(None)] * out.ndim
        seleccion[eje] = calcular_indices_gray(int(np.log2(N)))
        reordenado[tuple(seleccion)] = out
        out = reordenado
    return out


#deshacemos la permutacion Gray (mismos indices, aplicados hacia el otro lado) para volver al orden fisico (kx, ky)
def deshacer_orden_gray(probs_gray):
    out = probs_gray
    for eje, N in enumerate(probs_gray.shape):
        out = np.take(out, calcular_indices_gray(int(np.log2(N))), axis=eje)
    return out


#Genera el mapa de frecuencias objetivo (el espectro que luego se codifica en el circuito)
def crear_espectro_objetivo(forma, beta, k_corte=None, P0=0.05):
    ks = [np.fft.fftfreq(N, d=1/N) for N in forma]
    K = np.meshgrid(*ks, indexing='ij')
    k_abs = np.sqrt(sum(Ki**2 for Ki in K))
    if k_corte is None:
        k_corte = min(forma) / 2

    amplitudes = np.zeros_like(k_abs)
    #quitamos el componente DC (k=0), ahi no existe la ley de potencias
    nonzero = k_abs > 0
    #amplitud en ^(-beta/2) para que al elevar al cuadrado (probabilidad) quede en ^(-beta)
    amplitudes[nonzero] = k_abs[nonzero] ** (-beta / 2)
    amplitudes *= np.exp(-(k_abs / k_corte) ** 10)

    probs = amplitudes ** 2
    #normalizamos dejando hueco para el termino de continua P0
    probs = probs * (1.0 - P0) / probs.sum()
    probs[(0,) * len(forma)] = P0
    return probs, k_abs


#2) AJUSTE DEL ANSATZ Y CIRCUITO

#regresion ridge ponderada resuelta a mano (ecuaciones normales)
def ajuste_ridge_ponderado(X, y, pesos, alpha=1e-9):
    N, M = X.shape
    X_aug = np.empty((N, M + 1), dtype=X.dtype)
    X_aug[:, :-1] = X
    X_aug[:, -1] = 1.0

    #solución de minimos cuadrados
    X_T_W = X_aug.T * pesos[None, :]
    A = X_T_W @ X_aug
    b = X_T_W @ y
    I = np.eye(M + 1); I[-1, -1] = 0.0
    A += alpha * I
    try:
        coefs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coefs = np.linalg.lstsq(A, b, rcond=None)[0]
    #devuelve (bias, pesos)
    return coefs[-1], coefs[:-1]


#recorre el arbol de bisección y ajusta el ansatz lineal capa a capa a los angulos necesarios
#peso_exp: cuanto peso extra se le da en el ajuste a las ramas de mayor probabilidad (es el exponente del peso en la formula a minimizar). Con 0.5 es fiel al script del articulo de turbulencias; con valores en torno a 2 se consigue mejor resultado (se puede ver con PESO_COMPARACION)
def ajustar_capas_ansatz(probs_gray, n, cutoff=1e-25, peso_exp=2.4):
    probs_flat = probs_gray.flatten()
    params = []
    current_probs = probs_flat
    for layer_idx in range(n):
        n_features = layer_idx
        n_blocks = 2 ** n_features  #numero de ramas del arbol hasta esta capa
        remainder = current_probs.shape[0] // (n_blocks * 2)
        reshaped = current_probs.reshape(n_blocks, 2, remainder)
        block_probs = reshaped.sum(axis=2)
        p0, p1 = block_probs[:, 0], block_probs[:, 1]  #probabilidades de que el bit n_i sea 0 o 1
        w = p0 + p1  #masa total de probabilidad que llega a esa rama

        y_angles = np.zeros(n_blocks)
        valid_mask = w > cutoff  #un valor mínimo para evitar errores de división entre 0
        if np.any(valid_mask):
            ratio = np.clip(p0[valid_mask] / w[valid_mask], 0, 1)
            #angulo Ry que reparte correctamente la probabilidad entre bit=0 y bit=1 en esa rama
            y_angles[valid_mask] = 2 * np.arccos(np.sqrt(ratio))

        if not np.any(valid_mask):
            params.append((0.0, []))
            continue

        y_train, w_train = y_angles[valid_mask], w[valid_mask]
        indices = np.arange(n_blocks)[valid_mask]

        if n_features == 0:
            X_train = np.zeros((len(y_train), 0))
        else:
            #patron de bits previos de cada rama, es la matriz de features del ajuste lineal
            X_train = ((indices[:, None] & (1 << np.arange(n_features)[::-1])) > 0).astype(float)

        amp_weights = w_train ** peso_exp
        max_w = np.max(amp_weights)
        #normalizamos los pesos y se evitan posibles errores
        if max_w > 0:
            amp_weights = amp_weights / max_w

        if n_features == 0:
            bias, pesos = y_train[0], []
        elif len(y_train) <= 1:
            bias, pesos = y_train[0], np.zeros(n_features)
        else:
            #en caso correcto, se resuelve θ(bits) ≈ bias + sum (pesoⱼ·bit), haciendolo lineal para las 2^n ramas
            bias, pesos = ajuste_ridge_ponderado(X_train, y_train, amp_weights, alpha=1e-9)

        params.append((bias, pesos))
    return params


def numero_angulos(n):
    #n fases Rz + una fase ZZ por cada par de qubits
    return n + n * (n - 1) // 2


#construye el circuito: puertas de codificacion (Ry/CRy) + aleatorizacion de fase (Rz/CX-Rz-CX)
def construir_circuito_codificacion(n, params, angulos, weight_thresh=1e-3):
    qr = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qr)
    #capa 0 del arbol -> ultimo qubit fisico, capa 1 -> penultimo...
    ordered_qubits = [qr[i] for i in range(n - 1, -1, -1)]

    for layer_idx, (bias, pesos) in enumerate(params):
        # a cada qubit le asignamos una puerta Ry (bias)
        target_qubit = ordered_qubits[layer_idx]
        qc.ry(bias, target_qubit)
        for i, w in enumerate(pesos):
            if np.abs(w) > weight_thresh:
                #CRy con cada qubit anterior de peso significativo, monta el ansatz lineal en superposicion (así evitamos implementar puertas innecesarias)
                qc.cry(w, ordered_qubits[i], target_qubit)

    #aleatorizamos la fase mediante puertas Rz y CX-Rz-CX, dandoles la fase 'angulos' que genera nuestra semilla
    phi_rand = angulos[:n]
    for i in range(n):
        qc.rz(phi_rand[i], qr[i])

    idx = n
    for i in range(n):
        for j in range(i + 1, n):
            gamma_ij = angulos[idx]; idx += 1
            # CX-Rz-CX: fase tipo ZZ entre cada par de qubits, no toca las amplitudes
            qc.cx(qr[i], qr[j]); qc.rz(gamma_ij, qr[j]); qc.cx(qr[i], qr[j])
    return qc


#deshace la codificacion Gray para volver al orden binario normal de Hamming
def decodificar_gray_binario(qc, qubits):
    n = len(qubits)
    for j in range(n - 2, -1, -1):
        qc.cx(qubits[j + 1], qubits[j])


#reparte los qubits por eje: el eje x (el ultimo del array) va en los qubits mas bajos, despues el y
def qubits_por_eje(qc, forma):
    bloques, inicio = [], 0
    for b in reversed(bits_por_eje(forma)):
        bloques.append(qc.qubits[inicio:inicio + b])
        inicio += b
    return bloques


#monta el circuito completo: codificacion + decode + QFT, por separado en cada eje
def ensamblar_circuito_completo(forma, params, angulos):
    n = sum(bits_por_eje(forma))
    qc = construir_circuito_codificacion(n, params, angulos)
    ejes = qubits_por_eje(qc, forma)
    for qubits in ejes:
        decodificar_gray_binario(qc, qubits)
    #IFFT separable = IFFT_x (x) IFFT_y, por eso se aplica una QFT independiente por eje
    for qubits in ejes:
        qc.append(QFTGate(len(qubits)), qubits)
    return qc


#test de sanidad: comprueba que el decode Gray->binario por eje manda cada estado base a su sitio
def _comprobar_decodificacion_gray(forma):
    n = sum(bits_por_eje(forma))
    indices_gray = [calcular_indices_gray(b) for b in bits_por_eje(forma)]
    for idx_fisico in np.ndindex(*forma):
        idx_gray = np.ravel_multi_index(tuple(g[i] for g, i in zip(indices_gray, idx_fisico)), forma)
        qc = QuantumCircuit(n)
        for b in range(n):
            if (idx_gray >> b) & 1:
                qc.x(b)
        for qubits in qubits_por_eje(qc, forma):
            decodificar_gray_binario(qc, qubits)
        idx = int(np.argmax(np.abs(Statevector.from_instruction(qc).data)))
        esperado = np.ravel_multi_index(idx_fisico, forma)
        assert idx == esperado, f"Gray decode falló: {idx_fisico} -> idx={idx:0{n}b} esperado={esperado:0{n}b}"
    print(f"[OK] Gray->binario por eje decodifica correctamente los {2**n} estados base (forma {forma}).")


#test de correcion: comprueba que decode+QFT por eje coincide con np.fft.ifftn (salvo el factor sqrt(N))
def _comprobar_convencion_qft(forma, semilla=7):
    n = sum(bits_por_eje(forma))
    rng = np.random.default_rng(semilla)
    espectro_fisico = rng.normal(size=forma) + 1j * rng.normal(size=forma)
    espectro_fisico /= np.linalg.norm(espectro_fisico)

    sv = Statevector(reordenar_probs_gray(espectro_fisico).flatten())
    qc_dec = QuantumCircuit(n)
    ejes = qubits_por_eje(qc_dec, forma)
    for qubits in ejes:
        decodificar_gray_binario(qc_dec, qubits)
    for qubits in ejes:
        qc_dec.append(QFTGate(len(qubits)), qubits)

    psi = sv.evolve(qc_dec).data.reshape(forma)
    ratio = psi / np.fft.ifftn(espectro_fisico)
    assert np.allclose(ratio, ratio.flat[0], atol=1e-8), "decode+QFT por eje no coincide con np.fft.ifftn"
    assert np.isclose(abs(ratio.flat[0]), np.sqrt(np.prod(forma)), atol=1e-6)
    print(f"[OK] decode+QFT por eje == sqrt(N) * np.fft.ifftn  (forma {forma}).")


#3) SEMILLAS Y ANGULOS ALEATORIOS

#genera una semilla entera de 32 bits (por defecto) midiendo qubits en H, y de ahi se saca todo lo aleatorio
def generar_semilla_cuantica(num_bits=32):
    qc = QuantumCircuit(num_bits, num_bits)
    qc.h(range(num_bits))
    qc.measure(range(num_bits), range(num_bits))
    sim = AerSimulator()
    counts = sim.run(qc, shots=1).result().get_counts()
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2)


#genera angulos aleatorios creando una semilla de 32 (modificable) bits por angulo a traves de un registro auxiliar y puertas H
def generacion_angulos_rand(n, bits_per_angle=32):
    count = numero_angulos(n)
    total_bits = count * bits_per_angle

    qc = QuantumCircuit(total_bits, total_bits)
    qc.h(range(total_bits))
    qc.measure(range(total_bits), range(total_bits))
    #simulamos el circuito y realizamos la medida
    sim = AerSimulator()
    counts = sim.run(qc, shots=1).result().get_counts()
    bitstring = list(counts.keys())[0]
    chunks = [bitstring[i * bits_per_angle:(i + 1) * bits_per_angle] for i in range(count)]
    ints = np.array([int(c, 2) for c in chunks])
    #calculamos cada uno de los angulos normalizandolos y multiplicando por 2Pi
    return ints / (2 ** bits_per_angle) * 2 * np.pi


#devuelve los angulos de la fase de la muestra i, segun la fuente elegida en FUENTE_ANGULOS
def obtener_angulos(i, n, cfg):
    if cfg["FUENTE_ANGULOS"] == "angulos_cuanticos":
        return generacion_angulos_rand(n, cfg["BITS_POR_ANGULO"])
    if cfg["FUENTE_ANGULOS"] == "semilla_cuantica":
        semilla = generar_semilla_cuantica()
        print(f"  semilla cuántica usada en la muestra {i}: {semilla}")
    else:
        semilla = cfg["SEMILLA"] + i
    return np.random.default_rng(semilla).uniform(0, 2 * np.pi, numero_angulos(n))


#4) SIMULACION Y MEDIDA

#monta y simula el circuito completo, y se guarda unicamente la parte real del estado final (la IFFT de un espectro con fase aleatoria da resultados complejos)
def ejecutar_campo(forma, params, angulos, restar_media=True, informar=False):
    qc = ensamblar_circuito_completo(forma, params, angulos)
    sim = AerSimulator(method='statevector')
    qc_run = qc.copy()
    qc_run.save_statevector()
    qc_compiled = transpile(qc_run, sim, optimization_level=1)
    t0 = time.time()
    psi = sim.run(qc_compiled).result().get_statevector().data
    if informar:
        print(f"Circuito transpilado: {qc_compiled.size()} puertas, profundidad {qc_compiled.depth()}")
        print(f"Simulacion: {time.time()-t0:.3f}s")
    campo = np.real(psi).reshape(forma)
    return campo - campo.mean() if restar_media else campo


#medimos solo la parte de codificacion del circuito, SIN el gray_a_binario ni la QFT (no dan info de las frecuencias)
#la fase (Rz / CX-Rz-CX) tampoco importa aqui porque no cambia la probabilidad de medir cada estado (solo afecta a la fase relativa entre ellos), asi que se le pasan angulos a 0 sin que afecte al resultado.
def medir_distribucion_frecuencias(n, params, n_shots=200_000):
    qc_medida = construir_circuito_codificacion(n, params, np.zeros(numero_angulos(n)))
    qc_medida.measure_all()
    sim = AerSimulator()
    counts = sim.run(qc_medida, shots=n_shots).result().get_counts()

    #pasamos de "bitstring medido" a indice entero -> es directamente el indice dentro de probs_gray.flatten()
    emp_gray = np.zeros(2 ** n)
    for bitstring, c in counts.items():
        idx = int(bitstring.replace(' ', ''), 2)
        emp_gray[idx] = c / n_shots
    return emp_gray


#5) ANALISIS

#agrupa los puntos de la rejilla por distancia al centro (bins logaritmicos) y promedia su potencia (no es posible abarcar cada K por separado)
def promedio_espectro_logaritmico(kk, power, bins=30, kmin=1, kmax=None):
    if kmax is None:
        kmax = kk.max()
    mask = (kk >= kmin) & (kk <= kmax)
    kk, power = kk[mask], power[mask]
    edges = np.logspace(np.log10(kmin), np.log10(kmax), bins + 1)
    idx = np.digitize(kk, edges)
    centros, avg = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() > 0:
            centros.append(np.sqrt(edges[i - 1] * edges[i]))
            avg.append(power[m].mean())
    #devuelve los bins y su potencia promedio
    return np.array(centros), np.array(avg)


#calcula el R^2 entre dos ondas dadas
def calculo_r2(y_actual, y_ref):
    ss_res = np.sum((y_actual - y_ref) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rayleigh_test(angles):
    n = len(angles)
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R_bar = np.sqrt(C**2 + S**2) / n
    z = n * R_bar**2
    p = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288 * n**2))
    return R_bar, z, np.clip(p, 0, 1)


#se analiza el espectro radial de cada semilla y los resultados del r2 respecto de la media, la mediana y el caso puro
def analizar_espectro_y_r2(terreno_puro_x, fields_x, k_abs, beta, kmax_bin):
    def espectro_radial(field):
        power = np.abs(np.fft.fftn(field)) ** 2
        return promedio_espectro_logaritmico(k_abs.flatten(), power.flatten(), bins=25, kmin=1, kmax=kmax_bin)

    k_centers, power_pure = espectro_radial(terreno_puro_x)
    power_seeds = np.array([espectro_radial(f)[1] for f in fields_x])

    #calculamos la media y la mediana de las semillas
    power_seeds_mean = power_seeds.mean(axis=0)
    power_seeds_median = np.median(power_seeds, axis=0)
    power_theory = power_seeds_mean[0] * (k_centers / k_centers[0]) ** (-beta)

    #valores logaritmicos de la media y mediana, y de la funcion de potencia de cada semilla
    log_seeds = np.log(power_seeds)
    log_mean, log_median, log_pure = np.log(power_seeds_mean), np.log(power_seeds_median), np.log(power_pure)

    #cálculo de R² de la curva de potencia de cada semilla respecto de media, mediana y el caso puro
    return {
        'k_centers': k_centers, 'power_pure': power_pure, 'power_seeds': power_seeds,
        'power_seeds_mean': power_seeds_mean, 'power_seeds_std': power_seeds.std(axis=0),
        'power_seeds_median': power_seeds_median, 'power_theory': power_theory,
        'r2_seed_vs_mean': np.array([calculo_r2(l, log_mean) for l in log_seeds]),
        'r2_seed_vs_median': np.array([calculo_r2(l, log_median) for l in log_seeds]),
        'r2_seed_vs_pure': np.array([calculo_r2(l, log_pure) for l in log_seeds]),
        'r2_pure_vs_mean': calculo_r2(log_pure, log_mean),
        'r2_pure_vs_median': calculo_r2(log_pure, log_median),
    }


#6) FIGURAS

def guardar_figura(fig, nombre, cfg, **kwargs):
    if cfg["GUARDAR_FIGURAS"]:
        os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
        fig.savefig(os.path.join(BASE_DIR, 'figures', f'{nombre}_{cfg["DIMENSION"]}.png'), dpi=150, **kwargs)
    if not cfg["DIBUJAR"]:
        plt.close(fig)


#mapa de frecuencias objetivo: en 2D como mapa de calor y en 3D, en 1D como curva log-log
def figura_espectro_objetivo(probs, beta, cfg):
    if probs.ndim == 1:
        N = len(probs)
        k = np.arange(1, N // 2 + 1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog(k, probs[1:N // 2 + 1], color='tab:blue')
        ax.set_xlabel('|k|'); ax.set_ylabel('probabilidad')
        ax.set_title(f'Espectro objetivo de frecuencias (β={beta})')
    else:
        Ny, Nx = probs.shape
        kx_shift = np.fft.fftshift(np.fft.fftfreq(Nx, d=1 / Nx))
        ky_shift = np.fft.fftshift(np.fft.fftfreq(Ny, d=1 / Ny))
        #con rejillas no cuadradas el corte en k deja algunas esquinas a 0 exacto; se dibujan con la menor probabilidad no nula para evitar log10(0)
        Z = np.fft.fftshift(np.log10(np.maximum(probs, probs[probs > 0].min())))

        fig = plt.figure(figsize=(13, 5.5))
        ax = fig.add_subplot(1, 2, 1)
        im = ax.imshow(Z, cmap='viridis', origin='lower',
                       extent=[kx_shift.min(), kx_shift.max(), ky_shift.min(), ky_shift.max()])
        fig.colorbar(im, ax=ax, label='log10(probabilidad)')
        ax.set_title(f'Mapa de frecuencias (β={beta})')
        ax.set_xlabel('kx'); ax.set_ylabel('ky')

        ax3d = fig.add_subplot(1, 2, 2, projection='3d')
        KY_s, KX_s = np.meshgrid(ky_shift, kx_shift, indexing='ij')
        surf = ax3d.plot_surface(KX_s, KY_s, Z, cmap='viridis', linewidth=0, antialiased=True)
        ax3d.set_xlabel('kx'); ax3d.set_ylabel('ky'); ax3d.set_zlabel('log10(probabilidad)')
        ax3d.set_title(f'Mapa de frecuencias en 3D (β={beta})')
        fig.colorbar(surf, ax=ax3d, shrink=0.6, label='log10(probabilidad)')
    fig.tight_layout()
    guardar_figura(fig, 'espectro_objetivo', cfg)


#se comparan la medida cuantica real contra la teorica, una vez por cada peso que se quiera probar
def figura_medida_vs_teorica(probs_fisicas, k_abs, params_por_peso, n, kmax_bin, cfg):
    #bineado radial de la distribucion teorica (misma idea que el bineado del espectro del campo real. Agrupa por radio |k| en vez de casilla a casilla)
    k_centros, prob_teorica = promedio_espectro_logaritmico(k_abs.flatten(), probs_fisicas.flatten(), bins=25, kmin=1, kmax=kmax_bin)
    anchuras = np.append(np.diff(k_centros), np.diff(k_centros)[-1])
    # como en una escala lineal la ley de potencias se aplasta enseguida, se recorta el eje X hasta donde la probabilidad teorica todavia es al menos un 1% del pico. (modificable)
    k_visible = k_centros[prob_teorica >= prob_teorica.max() * 0.01]

    fig, axes = plt.subplots(1, len(params_por_peso), figsize=(6 * len(params_por_peso), 5), squeeze=False)
    for ax, (peso, params_x), color in zip(axes[0], params_por_peso, ['tab:red', 'tab:blue']):
        emp_gray = medir_distribucion_frecuencias(n, params_x, n_shots=cfg["N_SHOTS"]).reshape(probs_fisicas.shape)
        emp_fisica = deshacer_orden_gray(emp_gray)
        _, prob_medida = promedio_espectro_logaritmico(k_abs.flatten(), emp_fisica.flatten(), bins=25, kmin=1, kmax=kmax_bin)
        #con pocos shots, los bins de |k| alto pueden quedarse sin ninguna medida, y el log(0) daria nan; esos bins se dejan fuera de la correlacion
        validos = prob_medida > 0
        corr = np.corrcoef(np.log(prob_medida[validos]), np.log(prob_teorica[validos]))[0, 1]
        print(f"expo_peso= {peso:.2f}: correlación log-log medida vs teórica = {corr:.4f}"
              + (f"  ({(~validos).sum()} de {len(validos)} bins sin medidas, no se cuentan)" if (~validos).any() else ""))

        ax.bar(k_centros, prob_teorica, width=anchuras * 0.9, color='black', alpha=0.5, label='teórica')
        ax.bar(k_centros, prob_medida, width=anchuras * 0.9, color=color, alpha=0.5, label=f'medida ({cfg["N_SHOTS"]} shots)')
        ax.set_xlabel('|k|'); ax.set_ylabel('probabilidad'); ax.legend(fontsize=8)
        ax.set_title(f"expo_peso = {peso:.2f}  (correlación log-log = {corr:.3f})")
        ax.set_xlim(0, k_visible.max() * 1.1)
    fig.tight_layout()
    guardar_figura(fig, 'medida_vs_teorica', cfg)


#dibuja el circuito y lo guarda a fichero
def figura_circuito(qc, cfg):
    fig = qc.draw(output='mpl', fold=-1, scale=0.6)
    guardar_figura(fig, 'circuito', cfg, bbox_inches='tight')
    print(f"Circuito guardado en figures/circuito_{cfg['DIMENSION']}.png ({qc.num_qubits} qubits, {qc.size()} puertas).")


def figura_campo(campo, beta, cfg):
    if campo.ndim == 1:
        N = len(campo)
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(np.arange(N), campo / campo.std(), linewidth=0.5, color='tab:blue')
        ax.set_xlabel('Posición x (índice de grilla)')
        ax.set_ylabel('Ruido (normalizado)')
        ax.set_title(f'Ruido fBm cuántico-codificado: n={int(np.log2(N))} qubits, N={N}, β={beta}')
        ax.set_xlim(0, N)
    else:
        Ny, Nx = campo.shape
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(campo, cmap='terrain', origin='lower')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Terreno fBm 2D: {Nx}×{Ny}, β={beta}')
        fig.colorbar(im, ax=ax, shrink=0.8, label='altura')
    fig.tight_layout()
    guardar_figura(fig, 'campo', cfg)


#se comprueba que el resultado tiene las propiedades estadisticas correctas: pendiente del espectro (beta) y exponente de Hurst calculados vs esperados
def figura_validacion(campo, k_abs, beta, cfg):
    d = campo.ndim
    N = min(campo.shape)
    potencia = np.abs(np.fft.fftn(campo)) ** 2
    kk, pp = k_abs.flatten(), potencia.flatten()

    centros_k, potencia_bineada = promedio_espectro_logaritmico(kk, pp, bins=30, kmin=1, kmax=N / 4)
    pendiente, interseccion = np.polyfit(np.log(centros_k), np.log(potencia_bineada), 1)
    #se devuelve la pendiente (beta) calculada y la esperada
    print(f"Pendiente espectral ajustada: {pendiente:.3f}   (objetivo: -{beta:.2f})")

    #RMS de los incrementos a distintas separaciones r (a lo largo del eje x)
    separaciones = np.unique(np.logspace(0, np.log10(N // (8 if d == 1 else 4)), 25 if d == 1 else 20).astype(int))
    separaciones = separaciones[separaciones > 0]
    rms = np.array([np.sqrt(np.mean((campo - np.roll(campo, -r, axis=-1)) ** 2)) for r in separaciones])
    H_ajustada, _ = np.polyfit(np.log(separaciones), np.log(rms), 1)
    H_esperada = (beta - d) / 2
    #se calcula el exponente de Hurst y se compara con el esperado
    print(f"Exponente de Hurst ajustado H: {H_ajustada:.3f}   (esperado de β=2H+{d}: H={H_esperada:.3f})")

    fig, axes = plt.subplots(1, 3 if d == 1 else 2, figsize=(16 if d == 1 else 11, 4))
    mask = (kk > 1) & (kk < N / 4)
    axes[0].loglog(kk[mask], pp[mask], '.', alpha=0.1, color='tab:blue', markersize=2, label='periodograma crudo')
    axes[0].loglog(centros_k, potencia_bineada, 'o', color='tab:red', markersize=4, label='promedio bineado (log)')
    axes[0].loglog(centros_k, np.exp(interseccion) * centros_k ** pendiente, 'k-', label=f'ajuste: pendiente={pendiente:.2f}')
    axes[0].loglog(centros_k, potencia_bineada[0] * (centros_k / centros_k[0]) ** (-beta), '--', color='gray', label=f'β objetivo = {beta:.2f}')
    axes[0].set_xlabel('|k|'); axes[0].set_ylabel('Potencia'); axes[0].legend(fontsize=8)
    axes[0].set_title('Espectro de potencia' + (' radial' if d == 2 else ''))

    axes[1].loglog(separaciones, rms, 'o', color='tab:orange', markersize=4)
    axes[1].loglog(separaciones, rms[0] * (separaciones / separaciones[0]) ** H_ajustada, 'k-', label=f'ajuste: H={H_ajustada:.2f}')
    axes[1].set_xlabel('separación r' + (' (eje x)' if d == 2 else '')); axes[1].set_ylabel('RMS incremento')
    axes[1].legend(); axes[1].set_title('Escalado de incrementos (rugosidad)')

    if d == 1:
        #espectro de potencia en escala lineal, solo el primer 10% de N
        n_lin = int(0.10 * N)
        k_teoria = np.arange(1, n_lin)
        axes[2].plot(kk[:n_lin], pp[:n_lin], color='tab:blue', linewidth=0.9, label='potencia medida')
        axes[2].plot(k_teoria, pp[1] * k_teoria ** (-beta), '--', color='gray', linewidth=1.3, label=f'β objetivo = {beta:.2f}  (1/k^β)')
        axes[2].set_xlabel('k'); axes[2].set_ylabel('Potencia'); axes[2].legend(fontsize=9)
        axes[2].set_title(f'Escala lineal (k = 0..{n_lin}, primer 10% de N={N})')
    fig.tight_layout()
    guardar_figura(fig, 'validacion', cfg)
    return pendiente, H_ajustada


def theta_label(angles, n_show=3):
    preview = ", ".join(f"{a:.2f}" for a in angles[:n_show])
    return f"θ=[{preview}, ...]"


#dibujamos los terrenos: caso puro + cada semilla, con la misma escala de color para todos (si no la comparacion visual no seria justa)
#si hay PESO_COMPARACION, dos columnas -- izquierda peso_objetivo, derecha peso_comparacion -- para comparar en paralelo el mismo caso con ambos pesos
def figura_terrenos(casos_por_peso, angulos_todas, cfg):
    pesos = [p for p, _, _ in casos_por_peso]
    todos = np.concatenate([np.concatenate([[puro], campos]).ravel() for _, puro, campos in casos_por_peso])
    vmax = np.percentile(np.abs(todos), 99)
    etiquetas = ['puro\nθ=[0, 0, 0, ...]'] + [f'semilla {i}\n{theta_label(a)}' for i, a in enumerate(angulos_todas)]
    n_casos = len(etiquetas)

    #con un solo peso se reparten en una rejilla de 4 columnas; con dos, una fila por caso
    if len(pesos) == 1:
        ncols = min(4, n_casos)
        nrows = int(np.ceil(n_casos / ncols))
        celdas = [(i // ncols, i % ncols, 0, i) for i in range(n_casos)]
    else:
        ncols, nrows = 2, n_casos
        celdas = [(i, j, j, i) for i in range(n_casos) for j in range(2)]
    dim2 = casos_por_peso[0][1].ndim == 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, (4.3 if dim2 else 2.2) * nrows), squeeze=False)

    for fila, col, j_peso, i_caso in celdas:
        peso, puro, campos = casos_por_peso[j_peso]
        campo = puro if i_caso == 0 else campos[i_caso - 1]
        ax = axes[fila, col]
        if dim2:
            im = ax.imshow(campo, cmap='terrain', origin='lower', vmin=-vmax, vmax=vmax)
        else:
            ax.plot(campo, linewidth=0.5, color='tab:blue')
            ax.set_ylim(-vmax, vmax)
        titulo = etiquetas[i_caso].replace('\n', f' — expo_peso={peso:.2f}\n') if len(pesos) > 1 else etiquetas[i_caso]
        ax.set_title(titulo, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flatten()[len(celdas):]:
        ax.axis('off')
    if dim2:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='altura (misma escala en todas)')
    else:
        fig.tight_layout()
    guardar_figura(fig, 'terrenos_semillas', cfg, bbox_inches='tight')


#curva de potencia y R² de cada semilla, para cada peso
def figura_espectro_semillas(resultados_por_peso, beta, cfg):
    ncol = len(resultados_por_peso)
    fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 9), squeeze=False)
    for col, (peso, res) in enumerate(resultados_por_peso):
        etiqueta = f"expo_peso = {peso:.2f}"
        ax = axes[0, col]
        for p in res['power_seeds']:
            ax.loglog(res['k_centers'], p, color='tab:blue', alpha=0.15, linewidth=1)
        media, std = res['power_seeds_mean'], res['power_seeds_std']
        ax.fill_between(res['k_centers'], np.clip(media - std, media * 1e-3, None), media + std, color='tab:red', alpha=0.15, label='media ± desviación')
        ax.loglog(res['k_centers'], media, color='tab:red', linewidth=2, label='media de las semillas')
        ax.loglog(res['k_centers'], res['power_seeds_median'], color='tab:orange', linewidth=2, label='mediana de las semillas')
        ax.loglog(res['k_centers'], res['power_pure'], color='black', linewidth=2, label='puro (θ=0)')
        ax.loglog(res['k_centers'], res['power_theory'], '--', color='gray', linewidth=1.3, label=f'β objetivo = {beta}')
        ax.set_xlabel('|k|'); ax.set_ylabel('Potencia'); ax.legend(fontsize=7)
        ax.set_title(f'Espectro radial ({etiqueta})')

        ax2 = axes[1, col]
        x = np.arange(len(res['power_seeds']))
        w = 0.25
        ax2.bar(x - w, res['r2_seed_vs_mean'], width=w, color='tab:red', label='vs. media')
        ax2.bar(x, res['r2_seed_vs_median'], width=w, color='tab:orange', label='vs. mediana')
        ax2.bar(x + w, res['r2_seed_vs_pure'], width=w, color='tab:blue', label='vs. puro')
        ax2.set_xticks(x)
        ax2.set_xlabel('semilla'); ax2.set_ylabel('R²')
        ax2.set_title(f'R² de cada semilla ({etiqueta})')
        ax2.legend(fontsize=7)
    fig.tight_layout()
    guardar_figura(fig, 'espectro_semillas', cfg)


#distancia de cada semilla al caso puro (MSE y correlación de Pearson), y correlación media entre semillas distintas
def figura_estadisticas_semillas(puro, campos, cfg):
    mse_from_pure = np.array([np.mean((c - puro) ** 2) for c in campos])
    corr_from_pure = np.array([np.corrcoef(c.flatten(), puro.flatten())[0, 1] for c in campos])
    pair_corrs = np.array([np.corrcoef(campos[i].flatten(), campos[j].flatten())[0, 1]
                           for i in range(len(campos)) for j in range(i + 1, len(campos))])

    print(f"MSE(semilla, puro):         media={mse_from_pure.mean():.3e}  min={mse_from_pure.min():.3e}  max={mse_from_pure.max():.3e}")
    print(f"corr(semilla, puro):        media={corr_from_pure.mean():+.4f}  min={corr_from_pure.min():+.4f}  max={corr_from_pure.max():+.4f}")
    if len(pair_corrs):
        print(f"corr(semilla_i, semilla_j): media={pair_corrs.mean():+.4f}  std={pair_corrs.std():.4f}  (entre semillas distintas, deberia ser ~0)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(campos)), mse_from_pure, color='tab:blue')
    axes[0].set_xlabel('semilla'); axes[0].set_ylabel('MSE vs. caso puro'); axes[0].set_title('Distancia al caso puro (MSE)')
    axes[1].bar(range(len(campos)), corr_from_pure, color='tab:orange', label='vs. puro')
    if len(pair_corrs):
        axes[1].axhline(pair_corrs.mean(), color='k', linestyle='--', linewidth=1, label='media entre semillas')
    for ax in axes:
        ax.set_xticks(range(len(campos)))
    axes[1].set_xlabel('semilla'); axes[1].set_ylabel('correlación de Pearson'); axes[1].set_title('Correlación con el caso puro')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    guardar_figura(fig, 'estadisticas_semillas', cfg)


#graficamos los angulos finales de las N semillas para comprobar si su distribucion es homogenea
#la linea roja discontinua indica una distribución de angulos completamente homogenea. La idea de esto es validar visualmente cuanto se aleja nuestra distribución del caso ideal.
def figura_angulos(angulos_todas, cfg):
    angles_pool = np.asarray(angulos_todas).flatten()
    R_bar, z, p_rayleigh = rayleigh_test(angles_pool)
    print(f"Ángulos: {len(angles_pool)} en total, min={angles_pool.min():.4f}  max={angles_pool.max():.4f}  media={angles_pool.mean():.4f}  (esperado ~{np.pi:.4f})")
    conclusion = "no hay motivo para descartar que sean uniformes" if p_rayleigh >= 0.05 else "los ángulos podrían no ser uniformes"
    print(f"Test de Rayleigh: R={R_bar:.4f}, z={z:.3f}, p={p_rayleigh:.3f}  -> {conclusion}")

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection='polar')
    n_bins_rose = 24
    counts_rose, edges_rose = np.histogram(angles_pool, bins=n_bins_rose, range=(0, 2 * np.pi))
    width = edges_rose[1] - edges_rose[0]
    ax.bar(edges_rose[:-1], counts_rose, width=width, align='edge', color='tab:blue', edgecolor='white', alpha=0.85)
    ax.axhline(len(angles_pool) / n_bins_rose, color='tab:red', linestyle='--', linewidth=1.2)
    ax.set_title(f'Distribución de los {len(angles_pool)} ángulos medidos (p Rayleigh = {p_rayleigh:.2f})\n(línea roja = conteo esperado si uniforme)', fontsize=10, pad=20)
    fig.tight_layout()
    guardar_figura(fig, 'angulos', cfg, bbox_inches='tight')


#7) EJECUCION PRINCIPAL

def validar_configuracion(cfg):
    cfg = dict(cfg)
    cfg["DIMENSION"] = cfg["DIMENSION"].upper()
    if cfg["DIMENSION"] not in ("1D", "2D"):
        raise ValueError("DIMENSION tiene que ser '1D' o '2D'")
    if cfg["FUENTE_ANGULOS"] not in ("angulos_cuanticos", "semilla_cuantica", "clasica"):
        raise ValueError("FUENTE_ANGULOS tiene que ser 'angulos_cuanticos', 'semilla_cuantica' o 'clasica'")
    if cfg["N_SEMILLAS"] < 1:
        raise ValueError("N_SEMILLAS tiene que ser 1 o más")
    return cfg


def main(config=CONFIG):
    cfg = validar_configuracion(config)
    plt.close('all')
    dimension, beta = cfg["DIMENSION"], cfg["BETA"]

    #forma de la rejilla en el orden de los ejes del array: 1D (Nx,), 2D (Ny, Nx)
    bits = (cfg["NX"],) if dimension == "1D" else (cfg["NY"], cfg["NX"])
    forma = tuple(2 ** b for b in bits)
    n = sum(bits)
    kmax_bin = min(forma) / 2  #limite de radio |k| compartido por los bineados de la medida y de las semillas

    #tests de sanidad del orden Gray y de la QFT (con rejillas pequeñas: 1D de 2^5 y 2^6 puntos, 2D de 2^2 x 2^3 y 2^3 x 2^4)
    _comprobar_decodificacion_gray((2**5,) if dimension == "1D" else (2**2, 2**3))
    _comprobar_convencion_qft((2**6,) if dimension == "1D" else (2**3, 2**4))

    #1) espectro objetivo y ajuste del ansatz (con el peso objetivo y, si se pide, con el de comparacion)
    t0 = time.time()
    probs_fisicas, k_abs = crear_espectro_objetivo(forma, beta, k_corte=cfg["K_CORTE"], P0=cfg["P0"])
    probs_gray = reordenar_probs_gray(probs_fisicas)
    pesos = [cfg["PESO_OBJETIVO"]] + ([cfg["PESO_COMPARACION"]] if cfg["PESO_COMPARACION"] is not None else [])
    params_por_peso = [(p, ajustar_capas_ansatz(probs_gray, n, peso_exp=p)) for p in pesos]
    params = params_por_peso[0][1]
    print(f"Ajuste ridge (n={n} capas): {time.time()-t0:.3f}s")
    print(f"n={n} qubits, {numero_angulos(n)} angulos de fase por muestra, N_SEMILLAS={cfg['N_SEMILLAS']}")
    if cfg["DIBUJAR"] or cfg["GUARDAR_FIGURAS"]:
        figura_espectro_objetivo(probs_fisicas, beta, cfg)

    #2) medida estadistica del circuito vs distribucion teorica
    if cfg["MEDIR_VS_TEORICA"]:
        figura_medida_vs_teorica(probs_fisicas, k_abs, params_por_peso, n, kmax_bin, cfg)

    #3) angulos de las semillas y campos (semillas + caso puro con angulos a 0)
    angulos_todas = np.array([obtener_angulos(i, n, cfg) for i in range(cfg["N_SEMILLAS"])])
    campos = np.array([ejecutar_campo(forma, params, a, cfg["RESTAR_MEDIA"], informar=(i == 0))
                       for i, a in enumerate(angulos_todas)])
    for i, c in enumerate(campos):
        print(f"  semilla {i:2d}: std={c.std():.3e}")

    if cfg["DIBUJAR_CIRCUITO"]:
        figura_circuito(ensamblar_circuito_completo(forma, params, angulos_todas[0]), cfg)
    if cfg["DIBUJAR"] or cfg["GUARDAR_FIGURAS"]:
        figura_campo(campos[0], beta, cfg)

    #4) validacion de beta y Hurst sobre la semilla 0
    if cfg["VALIDAR_BETA_HURST"]:
        figura_validacion(campos[0], k_abs, beta, cfg)

    #5) comparacion de las semillas con el caso puro, para cada peso (mismas fases, asi la unica diferencia entre pesos es el ajuste ridge)
    terreno_puro = ejecutar_campo(forma, params, np.zeros(numero_angulos(n)), cfg["RESTAR_MEDIA"])
    if cfg["COMPARAR_SEMILLAS"]:
        casos_por_peso = [(pesos[0], terreno_puro, campos)]
        for peso, params_x in params_por_peso[1:]:
            puro_x = ejecutar_campo(forma, params_x, np.zeros(numero_angulos(n)), cfg["RESTAR_MEDIA"])
            campos_x = np.array([ejecutar_campo(forma, params_x, a, cfg["RESTAR_MEDIA"]) for a in angulos_todas])
            casos_por_peso.append((peso, puro_x, campos_x))

        resultados_por_peso = [(peso, analizar_espectro_y_r2(puro_x, campos_x, k_abs, beta, kmax_bin))
                               for peso, puro_x, campos_x in casos_por_peso]
        #printeamos los resultados
        for peso, res in resultados_por_peso:
            print(f"\n--- expo_peso = {peso:.2f} ---")
            print("R² de la curva de potencia de cada semilla respecto de la      media   -   mediana  -  puro")
            for i in range(len(campos)):
                print(f"  semilla {i:2d}:                              {res['r2_seed_vs_mean'][i]:7.4f}    {res['r2_seed_vs_median'][i]:7.4f}    {res['r2_seed_vs_pure'][i]:7.4f}")
            print(f"  -> media entre semillas:                    {res['r2_seed_vs_mean'].mean():7.4f}    {res['r2_seed_vs_median'].mean():7.4f}    {res['r2_seed_vs_pure'].mean():7.4f}")
            print(f"  R^2 del caso puro respecto de la media:    {res['r2_pure_vs_mean']:.4f}")
            print(f"  R^2 del caso puro respecto de la mediana:  {res['r2_pure_vs_median']:.4f}")
        print()

        if cfg["DIBUJAR"] or cfg["GUARDAR_FIGURAS"]:
            figura_terrenos(casos_por_peso, angulos_todas, cfg)
            figura_espectro_semillas(resultados_por_peso, beta, cfg)
            figura_estadisticas_semillas(terreno_puro, campos, cfg)

    #6) dispersion de los angulos
    if cfg["ANALIZAR_ANGULOS"]:
        figura_angulos(angulos_todas, cfg)

    #guardamos en disco unicamente la version 'buena' (peso_objetivo), la de peso_comparacion no hace falta
    if cfg["GUARDAR_DATOS"]:
        os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
        np.save(os.path.join(BASE_DIR, 'data', f'terreno_puro_{dimension}.npy'), terreno_puro)
        np.save(os.path.join(BASE_DIR, 'data', f'todas_las_semillas_{dimension}.npy'), campos)
        np.save(os.path.join(BASE_DIR, 'data', f'todos_los_angulos_{dimension}.npy'), angulos_todas)

    if cfg["DIBUJAR"]:
        plt.show()
    return campos, angulos_todas


if __name__ == "__main__":
    main()
