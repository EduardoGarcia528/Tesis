import os

def recolectar_tempos(ruta_base, nombre_archivo="tempo.txt", profundidad_max=8):
    lineas_unicas = set()

    for raiz, dirs, archivos in os.walk(ruta_base):
        # Calcular profundidad relativa desde la carpeta base
        profundidad = os.path.relpath(raiz, ruta_base).count(os.sep)
        if profundidad > profundidad_max:
            # No descender más en esta rama
            dirs[:] = []
            continue

        if nombre_archivo in archivos:
            ruta_completa = os.path.join(raiz, nombre_archivo)
            try:
                with open(ruta_completa, "r", encoding="utf-8") as f:
                    for linea in f:
                        linea = linea.strip()
                        if linea:
                            lineas_unicas.add(linea)
            except Exception as e:
                print(f"Error al leer {ruta_completa}: {e}")

    return sorted(lineas_unicas)

def guardar_lineas_en_archivo(lineas, archivo_salida="tempo_completo.txt"):
    try:
        with open(archivo_salida, "w", encoding="utf-8") as f:
            for linea in lineas:
                f.write(linea + "\n")
        print(f"Archivo guardado exitosamente: {archivo_salida}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

# ==== USO ====
# Cambia esta ruta a tu carpeta base
carpeta_raiz = "data\humdrum-data-numpy"

lineas = recolectar_tempos(carpeta_raiz)
guardar_lineas_en_archivo(lineas)
