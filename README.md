# Quantum Perlin Noise

**[English](#english) · [Español](#español)**

## English

Fractal Brownian (fBm) noise generated with a quantum circuit, plus a classical tool to study the spectrum of white, Perlin and fBm noise.

| File | Purpose |
|---|---|
| `generacion_fbm_cuantico.py` | Generates 1D signals or 2D terrains of fBm noise with a quantum circuit (simulated with Qiskit Aer) and checks that they have the right statistics. |
| `analisis_espectral_ruido.py` | Generates white, Perlin and fBm noise classically and analyses its Fourier spectrum: highest-weight coefficients, phases and spectral decay. |

### Requirements

Python 3 (tested with 3.13).

```bash
pip install numpy matplotlib qiskit qiskit-aer pylatexenc perlin-numpy
```

### Usage

Set the parameters in the `CONFIG` dictionary at the top of each script and run it:

```bash
python generacion_fbm_cuantico.py
python analisis_espectral_ruido.py
```

### How the quantum generator works

1. It builds the target spectrum P(k) ∝ 1/k^β, where β = 2H + d.
2. It encodes that spectrum in the amplitudes of the qubits (nx in 1D, nx + ny in 2D) with Ry/CRy gates in Gray order. The angles come from a layer-by-layer weighted ridge fit.
3. It adds a random phase with Rz and CX-Rz-CX gates. The angles come from a classical seed, from a 32-bit quantum seed, or from a separate quantum measurement for each angle.
4. It undoes the Gray order and applies a QFT on each axis, which acts as the inverse FFT. The real part of the final state is the noise.

It then checks the result: the measured β and Hurst exponent against the theoretical ones, a 200,000-shot measurement against the target distribution, several seeds against the case without random phase, and how uniform the angles are. Figures are saved in `figures/` and data in `data/`, next to the script.

### Author

Mikel Muñoz Vega

---

## Español

Ruido browniano fractal (fBm) generado con un circuito cuántico, más una herramienta clásica para estudiar el espectro del ruido blanco, Perlin y fBm.

| Archivo | Para qué sirve |
|---|---|
| `generacion_fbm_cuantico.py` | Genera señales 1D o terrenos 2D de ruido fBm con un circuito cuántico (simulado con Qiskit Aer) y comprueba que tienen la estadística correcta. |
| `analisis_espectral_ruido.py` | Genera ruido blanco, Perlin y fBm de forma clásica y analiza su espectro de Fourier: coeficientes de mayor peso, fases y caída del espectro. |

### Requisitos

Python 3 (probado con la 3.13).

```bash
pip install numpy matplotlib qiskit qiskit-aer pylatexenc perlin-numpy
```

### Uso

Ajusta los parámetros en el diccionario `CONFIG` del principio de cada script y ejecútalo:

```bash
python generacion_fbm_cuantico.py
python analisis_espectral_ruido.py
```

### Cómo funciona el generador cuántico

1. Construye el espectro objetivo P(k) ∝ 1/k^β, con β = 2H + d.
2. Codifica ese espectro en las amplitudes de los qubits (nx en 1D, nx + ny en 2D) con puertas Ry/CRy en orden Gray. Los ángulos salen de un ajuste ridge ponderado, capa a capa.
3. Añade una fase aleatoria con puertas Rz y CX-Rz-CX. Los ángulos salen de una semilla clásica, de una semilla cuántica de 32 bits o de una medida cuántica independiente para cada ángulo.
4. Deshace el orden Gray y aplica una QFT en cada eje, que hace de FFT inversa. La parte real del estado final es el ruido.

Después comprueba el resultado: β y exponente de Hurst medidos frente a los teóricos, una medida de 200 000 shots frente a la distribución objetivo, varias semillas frente al caso sin fase aleatoria y la uniformidad de los ángulos. Las figuras se guardan en `figures/` y los datos en `data/`, junto al script.

### Autor

Mikel Muñoz Vega
