package NeuralNetwork;

import java.util.function.Function;

/**
 * Enthält eine Auswahl verschiedener Aktivierungsfunktionen für neuronale
 * Netze.
 */
public enum ActivationFunction {

    /** Lineare Aktivierungsfunktion (keine Veränderung). */
    LINEAR(x -> x),

    /** Sigmoid-Funktion (bildet Werte auf den Bereich (0, 1) ab). */
    SIGMOID(x -> 1.0 / (1.0 + Math.exp(-x))),

    /** ReLU (Rectified Linear Unit) - gibt 0 für negative Werte zurück, sonst x. */
    RELU(x -> Math.max(0.0, x)),

    /** Tanh (Hyperbolischer Tangens) - bildet Werte auf den Bereich (-1, 1) ab. */
    TANH(Math::tanh);

    private final Function<Double, Double> function;

    ActivationFunction(Function<Double, Double> function) {
        this.function = function;
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
