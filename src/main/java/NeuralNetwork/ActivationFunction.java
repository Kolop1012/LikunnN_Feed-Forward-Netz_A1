package NeuralNetwork;

import java.util.function.Function;

/**
 * Enthält eine Auswahl verschiedener Aktivierungsfunktionen für neuronale
 * Netze.
 */
public enum ActivationFunction {

    /**
     * Lineare Aktivierungsfunktion (keine Veränderung).
     */
    @SuppressWarnings("unused")
    LINEAR(x -> x, x -> 1.0),

    /**
     * Sigmoid-Funktion (bildet Werte auf den Bereich (0, 1) ab).
     */
    @SuppressWarnings("unused")
    SIGMOID(x -> 1.0 / (1.0 + Math.exp(-x)), x -> {
        double s = 1.0 / (1.0 + Math.exp(-x));
        return s * (1.0 - s);
    }),

    /**
     * ReLU (Rectified Linear Unit) - gibt 0 für negative Werte zurück, sonst x.
     */
    @SuppressWarnings("unused")
    RELU(
            x -> Math.max(0.0, x),
            x -> x > 0.0 ? 1.0 : 0.0
    ),

    /**
     * Tanh (Hyperbolischer Tangens) - bildet Werte auf den Bereich (-1, 1) ab.
     */
    @SuppressWarnings("unused")
    TANH(
            Math::tanh,
            x -> {
                double t = Math.tanh(x);
                return 1.0 - (t * t);
            }
    );

    private final Function<Double, Double> function;
    private final Function<Double, Double> derivative;

    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    public double apply(double x) {
        return function.apply(x);
    }

    public double applyDerivative(double x) {
        return derivative.apply(x);
    }

    /**
     * Gibt die zugrundeliegende Funktion zurück.
     *
     * @return Die Funktionsreferenz.
     */
    public Function<Double, Double> getFunction() {
        return function;
    }
}
