import math
import pandas as pd
 
# 1. MEDIDAS DE IMPUREZA
 
def entropy(labels):
    """
    Entropy(S) = - sum( p_i * log2(p_i) )   para cada clase i
 
    Entropy = 0  -> todas las etiquetas son iguales (nodo puro)
    Entropy = 1  -> las clases estan perfectamente mezcladas (caso binario)
    """
    total = len(labels)
 
    # 1. Contar cuantos ejemplos hay de cada clase
    conteo_por_clase = {}
    for etiqueta in labels:
        conteo_por_clase[etiqueta] = conteo_por_clase.get(etiqueta, 0) + 1
 
    # 2. Sumar -p_i * log2(p_i) para cada clase
    ent = 0.0
    for clase, conteo in conteo_por_clase.items():
        p = conteo / total
        ent -= p * math.log2(p)
 
    return ent
  
# 2. SPLIT
def ganancia_de_informacion(y_padre, y_izquierda, y_derecha):
    """
    Gain(S, split) = Entropy(S) - suma_ponderada( Entropy(rama) )
    """
    n = len(y_padre)
    n_izq = len(y_izquierda)
    n_der = len(y_derecha)
 
    # Si alguna rama queda vacia, esta division no sirve
    if n_izq == 0 or n_der == 0:
        return 0.0
 
    impureza_padre = entropy(y_padre)
    impureza_izq = entropy(y_izquierda)
    impureza_der = entropy(y_derecha)
 
    impureza_ponderada = (n_izq / n) * impureza_izq + (n_der / n) * impureza_der
 
    return impureza_padre - impureza_ponderada
 
 
# 3. BUSQUEDA DEL MEJOR SPLIT EN UN NODO
 
def es_columna_numerica(serie):
    tipos_numericos = ["int64", "int32", "float64", "float32"]
    return str(serie.dtype) in tipos_numericos
 
 
def encontrar_mejor_split(X, y, columnas):
    """
    Prueba todos los splits posibles en todas las columnas y
    regresa el que da la mayor ganancia de informacion.
    """
    mejor_ganancia = 0.0
    mejor_columna = None
    mejor_valor = None
    mejor_tipo = None
 
    for columna in columnas:
        valores_columna = X[columna]
 
        if es_columna_numerica(valores_columna):
            # --- caso numerico: probar umbrales ---
            valores_unicos = sorted(valores_columna.unique())
 
            for i in range(len(valores_unicos) - 1):
                umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
 
                mascara_izq = valores_columna <= umbral
                mascara_der = ~mascara_izq
 
                y_izq = y[mascara_izq]
                y_der = y[mascara_der]
 
                ganancia = ganancia_de_informacion(y, y_izq, y_der)
 
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_columna = columna
                    mejor_valor = umbral
                    mejor_tipo = "numerico"
 
        else:
            # --- caso categorico: probar "== valor" ---
            valores_unicos = valores_columna.unique()
 
            for valor in valores_unicos:
                mascara_izq = valores_columna == valor
                mascara_der = ~mascara_izq
 
                y_izq = y[mascara_izq]
                y_der = y[mascara_der]
 
                ganancia = ganancia_de_informacion(y, y_izq, y_der)
 
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_columna = columna
                    mejor_valor = valor
                    mejor_tipo = "categorico"
 
    return mejor_columna, mejor_valor, mejor_tipo, mejor_ganancia
 
 
# 4. CLASE ARBOL DE DECISION
def clase_mayoritaria(y):
    """Regresa la clase que mas se repite en y (para hojas)."""
    conteo_por_clase = {}
    for etiqueta in y:
        conteo_por_clase[etiqueta] = conteo_por_clase.get(etiqueta, 0) + 1
 
    # buscar manualmente cual clase tiene el conteo mas alto
    mejor_clase = None
    mejor_conteo = -1
    for clase, conteo in conteo_por_clase.items():
        if conteo > mejor_conteo:
            mejor_conteo = conteo
            mejor_clase = clase
 
    return mejor_clase
 
 
class DecisionTree:
    """
    Parametros:
    max_depth : profundidad maxima del arbol
    min_samples_split : minimo de ejemplos necesarios para dividir un nodo
    """
 
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.raiz = None  # aqui se guarda el arbol ya entrenado
 
    def fit(self, X, y):
        """Entrena el arbol recursivamente a partir de X (DataFrame) y y (Series)."""
        columnas = list(X.columns)
        self.raiz = self._construir_arbol(X, y, columnas, profundidad=0)
        return self
 
    def _construir_arbol(self, X, y, columnas, profundidad):
 
        # --- condiciones de paro ---
        # 1. Todos los ejemplos son de la misma clase -> nodo puro
        if len(set(y)) == 1:
            return {"hoja": True, "prediccion": y.iloc[0]}
 
        # 2. Se alcanzo la profundidad maxima
        if profundidad >= self.max_depth:
            return {"hoja": True, "prediccion": clase_mayoritaria(y)}
 
        # --- buscar el mejor split posible ---
        columna, valor, tipo, ganancia = encontrar_mejor_split(X, y, columnas)
 
        # --- dividir los datos segun el mejor split encontrado ---
        if tipo == "numerico":
            mascara_izq = X[columna] <= valor
        else:  # categorico
            mascara_izq = X[columna] == valor
        mascara_der = ~mascara_izq
 
        X_izq, y_izq = X[mascara_izq], y[mascara_izq]
        X_der, y_der = X[mascara_der], y[mascara_der]
 
        subarbol_izq = self._construir_arbol(X_izq, y_izq, columnas, profundidad + 1)
        subarbol_der = self._construir_arbol(X_der, y_der, columnas, profundidad + 1)
 
        return {
            "hoja": False,
            "columna": columna,
            "tipo": tipo,
            "valor": valor,
            "ganancia": ganancia,
            "izquierda": subarbol_izq,
            "derecha": subarbol_der,
        }
 
    def _predecir_una_fila(self, nodo, fila):
        """Recorre el arbol desde la raiz hasta una hoja para una sola fila."""
        if nodo["hoja"]:
            return nodo["prediccion"]
 
        if nodo["tipo"] == "numerico":
            if fila[nodo["columna"]] <= nodo["valor"]:
                return self._predecir_una_fila(nodo["izquierda"], fila)
            else:
                return self._predecir_una_fila(nodo["derecha"], fila)
        else:  # categorico
            if fila[nodo["columna"]] == nodo["valor"]:
                return self._predecir_una_fila(nodo["izquierda"], fila)
            else:
                return self._predecir_una_fila(nodo["derecha"], fila)
 
    def predict(self, X):
        """Predice la clase para cada fila de X (DataFrame). Regresa una lista."""
        predicciones = []
        for i in range(len(X)):
            fila = X.iloc[i]
            predicciones.append(self._predecir_una_fila(self.raiz, fila))
        return predicciones
 
    def imprimir_arbol(self, nodo=None, prefijo=""):
        """
        Imprime el arbol en texto, de forma legible, para revisar
        visualmente que las reglas aprendidas tengan sentido.
        """
        if nodo is None:
            nodo = self.raiz
 
        if nodo["hoja"]:
            print(prefijo + f"-> Prediccion: {nodo['prediccion']}")
            return
 
        if nodo["tipo"] == "numerico":
            condicion = f"{nodo['columna']} <= {nodo['valor']:.2f}"
        else:
            condicion = f"{nodo['columna']} == {nodo['valor']}"
 
        print(prefijo + f"[{condicion}]  (ganancia={nodo['ganancia']:.4f})")
        print(prefijo + " Si:")
        self.imprimir_arbol(nodo["izquierda"], prefijo + "   ")
        print(prefijo + " No:")
        self.imprimir_arbol(nodo["derecha"], prefijo + "   ")
  
# 5. AUTOPRUEBA CON "PLAY TENNIS"

if __name__ == "__main__":
 
    datos = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
                    "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast",
                    "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                         "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal",
                     "Normal", "High", "Normal", "Normal", "Normal", "High",
                     "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                 "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No",
                        "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
    }
 
    df = pd.DataFrame(datos)
    X = df.drop(columns=["PlayTennis"])
    y = df["PlayTennis"]
 
    print("=" * 55)
    print(" AUTOPRUEBA - dataset Play Tennis")
    print("=" * 55)
 
    # Verificacion manual del primer split (debe elegir Outlook como
    # raiz, igual que en la diapositiva vista en clase)
    columna, valor, tipo, ganancia = encontrar_mejor_split(X, y, list(X.columns))
    print("\nMejor split en la raiz:")
    print(f"  columna esperada : Outlook")
    print(f"  columna obtenida : {columna}")
    print(f"  ganancia obtenida: {ganancia:.3f}  (esperado ~0.246)")
 
    # Entrenar el arbol completo y mostrarlo
    arbol = DecisionTree(max_depth=5, min_samples_split=1)
    arbol.fit(X, y)
 
    print("\n" + "-" * 55)
    print(" Arbol entrenado")
    print("-" * 55)
    arbol.imprimir_arbol()
 
    # Evaluar sobre los mismos datos de entrenamiento
    predicciones = arbol.predict(X)
    aciertos = sum(p == real for p, real in zip(predicciones, y))
    total = len(y)
 
    print("\n" + "-" * 55)
    print(" Resultado")
    print("-" * 55)
    print(f"  Aciertos : {aciertos}/{total}")
    print(f"  Accuracy : {aciertos / total:.1%}")
    print("=" * 55)
