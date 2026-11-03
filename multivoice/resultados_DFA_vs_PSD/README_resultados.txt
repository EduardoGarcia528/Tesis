ETAPA 1 - COMPARACION DFA / PSD
=================================

Numero total de voces: 464
Ajustes definidos y con suficientes puntos (N/4): 380
Criterio conservador de rango y estabilidad (N/8): 65

Interpretacion de las etiquetas:
  CONSERVADOR: rango log suficiente en N/8, puntos minimos y
               alpha/beta estables al variar la escala maxima.
  EXPLORATORIO: ajuste N/4 disponible pero no supera los criterios.
  INSUFICIENTE: faltan puntos para ajustar DFA y/o PSD.
Esto NO certifica una ley de potencia ni estacionariedad.

Criterios: min decadas N/8=0.8; min escalas DFA=8;
min bins PSD=8; min frecuencias PSD=16;
|Delta alpha| <= 0.15;
|Delta beta| <= 0.25

El R² y el error son de los ajustes en escala log10-log10.
El error estandar OLS es solo orientativo: las estimaciones
a distintas escalas y bins espectrales no son independientes.

Estimables N/4 (n=380):
  mediana(delta) = -0.1368
  mediana(|delta|) = 0.2273
  Pearson(beta_pred,beta) = 0.7580
  Spearman(beta_pred,beta) = 0.7530

Criterio conservador (n=65):
  mediana(delta) = -0.1085
  mediana(|delta|) = 0.1847
  Pearson(beta_pred,beta) = 0.9164
  Spearman(beta_pred,beta) = 0.9187

Interpretacion estadistica:
PSD: metodo=welch; ventana=hann; detrend=linear.
La coincidencia beta = 2alpha-1 es una hipotesis teorica CON SUPUESTOS,
no una igualdad obligatoria para series cortas ni no estacionarias.
R² alto NO demuestra leyes de escala ni memoria de largo alcance.
Se trabaja en indice de evento (nota), no en tiempo fisico.
La bandera 'conservador' es un filtro exploratorio, NO un test formal.
