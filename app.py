from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from scipy.optimize import curve_fit

app = Flask(__name__)

def generate_plot():
    # Datos
    t = np.array([15, 30, 45, 60, 75, 90, 105, 120])
    y = np.array([0.2, 1.1, 2.9, 6.5, 13.2, 20.0, 26.1, 28.3])

    # Interpolación de Lagrange
    lagrange_poly = lagrange(t, y)
    y_lagrange = lagrange_poly(t)

    # Interpolación por Splines cúbicos
    cs = CubicSpline(t, y)
    t_smooth = np.linspace(15, 120, 500)
    y_spline = cs(t_smooth)

    # Regresión logística
    def logistic(t, a, b):
        H = 30
        return H / (1 + np.exp(-(a + b * t)))

    params, _ = curve_fit(logistic, t, y, p0=[-10, 0.1])
    a_fit, b_fit = params
    y_logistic = logistic(t_smooth, a_fit, b_fit)

    # Evaluaciones
    t_eval = 100
    y_lagrange_eval = lagrange_poly(t_eval)
    y_spline_eval = cs(t_eval)
    y_logistic_eval = logistic(t_eval, a_fit, b_fit)

    # Gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'o', label="Datos reales", color='black')
    plt.plot(t_smooth, y_spline, label="Splines cúbicos", linestyle='--')
    plt.plot(t_smooth, y_logistic, label="Regresión logística", linestyle='-')
    plt.plot(t, y_lagrange, label="Interpolación Lagrange", linestyle='dotted')
    plt.title("Crecimiento de la planta de zanahoria")
    plt.xlabel("Días después de siembra")
    plt.ylabel("Peso seco (g/planta)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Guardar la gráfica
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()

    # Preparar datos para la tabla
    table_data = {
        'days': [int(ti) for ti in t],
        'weights': [f"{wi:.1f}" for wi in y]
    }

    return {
        'lagrange': f"{y_lagrange_eval:.2f}",
        'spline': f"{y_spline_eval:.2f}",
        'logistic': f"{y_logistic_eval:.2f}",
        'plot_path': plot_path,
        'table_data': table_data
    }

@app.route('/')
def index():
    data = generate_plot()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
