# 🧠🔀 Predicciones a través de redes neuronales artificiales

Inspirada en la naturaleza de las redes neuronales de nuestro cerebro, en los siguientes proyectos se realizan predicciones tomando como entrada los datos de interés. El algoritmo implementado en cada proyecto es organizado en diferentes números de capas que representan ciertas funciones matemáticas. En cada capa se reconocen patrones y características que permiten, en última instancia, entregar una predicción con un cierto nivel de confianza. 

Los proyectos son presentados de acuerdo al orden de complejidad de la red neuronal y se ordenan de la siguiente manera:

- 1️⃣ Implementación de un perceptrón (red neuronal básica) para un problema de clasificación binaria: discernir entre especies de flores.

- 2️⃣ Diseño, entrenamiento y evaluación de una red neuronal multicapa para un problema de regresión: predecir precios de viviendas en Boston.

- 3️⃣ Diseño, entrenamiento y evaluación de una red neuronal multicapa para un problema de clasificación: identificar clientes potenciales. 

- 4️⃣ Introducción al Deep Learning: diseño, entrenamiento y evaluación de una red neuronal para determinar el precio justo para venta de vehículos usados. 


## 🌟 Funcionamiento

El algoritmo de una red neuronal funciona de acuerdo los siguientes pasos:

- 1️⃣ Inicialización de los pesos sinápticos.
- 2️⃣ Definición de la función de activación para abstraer relaciones no lineales. 
- 3️⃣ Aplicar funciones de costo para medir la diferencia entre el valor predicho y el valor real.
- 4️⃣ Aplicar el algoritmo de optimización para reducir el error o función de costo.
- 5️⃣ Ajustar los pesos sinápticos de acuerdo al paso 4.

### 🔻 Inicialización de los pesos

Los rangos de los pesos de inicialización que han demostrado ser más efectivos y que son utilizados en este proyecto, están definidos de acuerdo a las variaciones de Glorot/Xavier: 

- Intervalo Uniforme $[-x,x]$:

$$ x = \sqrt{  \frac{6}{E + S} } $$

- Intervalo Normal: con Media 0 y $\sigma$:

$$ \sigma = \sqrt{  \frac{2}{E + S} } $$

donde $E$ y $S$ son la cantidad de entradas y salidas. 

### 🔻 Funciones de activación

Algunas de las funciones más utilizadas son: 

- Función Sigmoid o Logística: Es utilizada especialmente para los modelos en los que tenemos que predecir la probabilidad como un resultado ya que tiene una salida multivalor acotada de (0,1). 

$$ f(x) = \frac{1}{1+e^x}$$ 

- Función Tangente Hiperbólica: es usualmente usada en problemas de clasificación binaria, con una salida acotada entre (-1,1).

$$ f(x) = \frac{2}{1+e^{2x}} - 1 $$

- ReLU (Rectified Linear Unit): tiene salida no acotada (0,∞) y derivadas positivas, por lo que es importante considerar las variables de entrada normalizadas (de 0 a 1). Es usada en problemas de regresión en los que se entrega una número final. 

$$  f(x) =  \begin{cases} 0 & \text{if } x < 0, x & \text{if } x > 0 \end{cases} \} $$


### 🔻 Funciones de costo

- Para variables numéricas: 

    - Mean Absolute Error: de fácil interpretación y robusta a outliers. 

    $$ MAE = \frac{1}{k} \sum_i^k |Real - Predicho| $$

    - Mean Squared Error: penaliza el modelo cuando existen grandes errores y es sensible a outliers.

    $$ MSE = \frac{1}{k} \sum_i^k (Real - Predicho)^2 $$

    - Mean Absolute Percentage Error: penaliza el modelo cuando existen grandes errores y es robusta a outliers. 
     $$ MAPE = \frac{1}{k} \sum_i^k | \frac{Real - Predicho}{Real}| $$

- Para variables categóricas:

  - Binary Cross-Entropy: Penaliza el modelo cuando existen grandes errores. En la siguiente ecuación, $y_i'$ es el valor predicho y $y_i$ el valor real. 

  $$ H_{y'}(y) := - \sum_{i} ({y_i' \log(y_i) + (1-y_i') \log (1-y_i)}) $$ 

  - Categorical Cross-Entropy: Usada en problemas multiclase, de igual manera penaliza el modelo cuando presenta grandes errores. En la siguiente ecuación $p(x)$ es el valor real, y $q(x)$ el valor predicho. 

  $$ H(p,q) = -\sum_{\forall x} p(x) \log(q(x)) $$

### 🔻 Algoritmos de optimización

- Gradiente descendiente: Calcula cómo deben ser alterados los pesos para que la función de coste pueda alcanzar un mínimo, la desventaja es que este cálculo se aplica sobre todo el dataset, por lo que toma mucho coste computacional y es posible que el gradiente se ubique solo un mínimo local.

- Gradiente descendiente estocástico (SGD): es una variación del gradiente descendiente en donde los pesos son modificados en cada lote (`batch`) de información. Presenta la ventaja de tener un coste computacional menor pero a cambio, los parámetros del modelo pueden tener gran varianza debido a la frecuencia de la actualización de los pesos.  

- Momentum: fue diseñado para reducir la varianza en del SGD, al acelerar la convergencia hacia la dirección relevante y reducir la fluctuación en la dirección irrelevante. 

- Gradiente adaptativo (AdaGrad): sigue el mismo principio que el algoritmo SGD, solo que las actualizaciones de los pesos es independiente uno del otro. Esto implica que cada peso empezará a obtener su valor y sucesivamente en algún momento en el tiempo se encuentra la solución global.

- ADAM (AdaGrad+Momentum): incorpora los principios anteriores para acelerar el descenso del gradiente al considerar pasos pequeños pero directos hacia la dirección correspondiente al mínimo.

## ✅ Evaluación

El algorimo aplicado en cada red neuronal representa un modelo de los datos, para determinar si el modelo logra generalizar su comportamiento se deben considerar:

- 1️⃣ Métricas de desempeño.
- 2️⃣ Curvas de aprendizaje.


### 🔻 Métricas de desempeño

- Para problemas de regresión: como en estos problemas la variable de salida es de tipo numérico, se pueden aplicar las mismas funciones de costo que se han definido anteriormente, es decir, `MAE`, `MSE` o `MAPE`, por mencionar algunos ejemplos. 

- Para problemas de clasifiación: en estos casos el modelo arroja un resultado Positivo o Negativo, que luego será evaluado como Verdadero o Falso dependiendo de si la predicción sea correcta o no. Los posibles resultados de un problema de clasificación se etiquetan como: 

  - $VP$: verdaderos positivos, 
  - $VN$: verdaderos negativos, 
  - $FP$: falsos positivos,

  - $FN$: falsos negativos,

  A partir de este conteo se definen las siguientes métricas: 

  - Accuracy: $$ \frac{VP+VN}{VP+VN+FP+FN} $$
  - Precision: $$ \frac{VP}{VP+FP} $$
  - Recall (o sensibilidad): $$ \frac{VP}{VP+FN} $$
  - F1-score: $$ \frac{ 2  \cdot (Recall \cdot Precision)}{(Recall + Precision)}$$

  Las métricas de desempeño en las clasificaciones se escogen dependiendo del contexto del problema a resolver.

### 🔻 Curvas de aprendizaje

A partir de las métricas descritas anteriormente, se construyen las curvas de aprendizaje. En ellas podemos evaluar si el algoritmo logró generalizar los datos, o por el contrario, necesita añadirle complejidad o reducirla. Estas características son apreciadas al graficar la métrica escogida en función del número de iteraciones (o épocas), donde podemos encontrarnos con los siguientes escenarios: 

- Underfitting (o sub ajuste): cuando el modelo es incapaz de obtener resultados correctos por falta de entrenamiento o de más muestras. Se reconoce visualmente cuando existe altos valores de pérdida en el set de entrenamiento y validación. 

- Overfitting (o sobre ajuste): cuando el modelo se ajusta solo a los datos de entrenamiento y se vuelve incapaz de reconocer nuevos datos. Es visualmente reconocido cuando existe una separación importante entre las pérdidas del set de entrenamiento y de validación. 

- Optimal fit: cuando el modelo logra captar el comportamiento general de los datos. Se puede identificar al obtener valores de pérdidas similares en los datos de entrenamiento y validación. 

<p align="center">
  <img width="700" src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff80030c0-6072-4d13-a6d7-e148f6c5c39a%2FUntitled.png?table=block&id=4c389bbb-7692-43aa-b4e8-016b1a14d03b&width=2050&userId=5d54f20c-b387-4e7e-b2a5-d88e235ada88&cache=v2">
</p>

La forma general de resolver el problema de underfitting es añadiendo complejidad a la red. Esto puede llevarse a cabo al incluir más datos o al aumentar el entrenamiento de la red. Para el problema del overfitting es recomendable reducir la cantidad de variables o añadir [técnicas de regularización](https://www.notion.so/mariajosemv/Redes-neuronales-en-Keras-y-ScikitLearn-b8fcf479b0464021bb85d1b2a8863404#150430fc557048d7820017da9ca66ea5), las cuales consisten en en disminuir la complejidad del modelo por medio de una penalización aplicada a sus variables más irrelevantes. 

----

## 📌 Notas 

Los proyectos fueron construidos utilizando las librerías [Scikit-Learn](https://scikit-learn.org/) y [Keras](https://keras.io/), ambos considerados frameworks de alto nivel, orientados a la experiencia de usuario y caracterizados por permitir realizar experimentaciones rápidas. 

Cada proyecto fue realizado como parte del [Curso de Redes Neuronales en Keras y Scikit-Learn](https://platzi.com/cursos/keras-neural-networks/) de Platzi. Para más información y detalles, te invito a leer mis notas del curso en [Notion](https://www.notion.so/mariajosemv/Redes-neuronales-en-Keras-y-ScikitLearn-b8fcf479b0464021bb85d1b2a8863404). ✨
