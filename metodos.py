import math
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return ((2*x)/math.exp(x**2)) - 2*x*y

def y_real(x, c):
    return (x**2+c)/math.exp(x**2)

def metodo_euler(funcion, x0, y0, a, b, n):
    resultados_x = [x0]
    resultados_y = [y0]
    h = (b-a)/n  
    for i in range(1, n+1):
        y0 = y0 + h * funcion(x0, y0)
        x0 = x0 + h
        resultados_x.append(x0)
        resultados_y.append(y0)
    return resultados_x, resultados_y

def predictor_Corrector(f, y0, a, b, n):
    x0 = a
    resultados_x = [x0]
    resultados_y = [y0]
    h = (b-a)/n
    for _ in range(1, n+1):
        # Paso de predicción usando el método de Euler
        y_pred = y0 + h * f(x0, y0)    
        # Paso de corrección usando la pendiente promedio
        pendiente_promedio = (f(x0, y0) + f(x0 + h, y_pred)) / 2.0
        y0 = y0 + h * pendiente_promedio    
        x0 = x0 + h
        resultados_x.append(x0)
        resultados_y.append(y0)
    return resultados_x, resultados_y

def metodo_runge_kutta2(funcion, x0, y0, a, b, n):
    resultados_x = [x0]
    resultados_y = [y0]
    h = (b-a)/n  
    for i in range(1, n+1):
        y0 = y0 + h * funcion(x0 + h/2, y0 + (h/2)*funcion(x0, y0))
        x0 = x0 + h
        resultados_x.append(x0)
        resultados_y.append(y0)
    return resultados_x, resultados_y

def metodo_runge_kutta4 (f, x0 ,y0 ,x1 ,n) :
    h = ( x1 - x0 )/n 
    xi = np.zeros (n+1) 
    yi = np.zeros (n+1) 
    xi [0]= x0
    yi [0]= y0
    for i in range (n) :
        k1 = f(xi[i], yi[i])
        k2 = f(xi[i] + h/2, yi[i] + k1 * h/2)
        k3 = f(xi[i] + h/2, yi[i] + k2 * h/2)
        k4 = f(xi[i] + h, yi[i] + k3 * h )
        yi[i+1]= yi[i] + h * (k1/6 + k2/3 + k3/3 + k4/6)
        xi[i+1]= xi[i]+ h
    return xi , yi 


# Condiciones iniciales
x_inicial = 0
extremo_inferior = 0
extremo_superior = 1.5
#Las claves del diccionario son las condiciones iniciales de y
#Los valores del diccionario son los valores reales de y en el extremo superior (el ultimo argumento vuelve a ser el valor de y en la condicion inicial de x)
y_iniciales = {0: y_real(extremo_superior, 0),
            1: y_real(extremo_superior, 1),
            -1: y_real(extremo_superior, -1)
            }


# Número de iteraciones
N = 20

def calcular_valores(resultados_x, resultados_y, valor_real, graficar):
    print(f"x = {resultados_x[N]}, y = {resultados_y[N]}, error = {abs(resultados_y[N] - valor_real)}")
    if graficar:
        plt.plot(resultados_x, resultados_y)   


for clave, valor in y_iniciales.items():
    print('\n')
    print('Metodo de Euler:')
    # Aplicar el método de Euler
    resultados_euler_x, resultados_euler_y = metodo_euler(f, x_inicial, clave, extremo_inferior, extremo_superior, N)
    # Imprimir resultados
    calcular_valores(resultados_euler_x, resultados_euler_y, valor, True)

    N = 10
    print('Metodo Predictor Corrector:')
    # Aplicar el método de Predictor Corrector
    resultados_predictor_x, resultados_predictor_y = predictor_Corrector(f, clave, extremo_inferior, extremo_superior, N)
    # Imprimir resultados
    calcular_valores(resultados_predictor_x, resultados_predictor_y, valor, False)

    print('Metodo Runge-Kutta de orden 2:')
    # Aplicar el método runge-kutta
    resultados_runge2_x, resultados_runge2_y = metodo_runge_kutta2(f, x_inicial, clave, extremo_inferior, extremo_superior, N)
    # Imprimir resultados
    calcular_valores(resultados_runge2_x, resultados_runge2_y, valor, False)

    N = 5
    print('Metodo Runge-Kutta de orden 4:')
    # Aplicar el método de Predictor Corrector
    resultados_runge4_x, resultados_runge4_y = metodo_runge_kutta4(f, extremo_inferior, clave, extremo_superior, N)
    # Imprimir resultados
    calcular_valores(resultados_runge4_x, resultados_runge4_y, valor, False)

    
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución de la Ecuación Diferencial')
plt.grid(True)
plt.show()