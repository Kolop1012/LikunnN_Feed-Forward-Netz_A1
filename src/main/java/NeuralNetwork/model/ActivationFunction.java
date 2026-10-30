package NeuralNetwork.model;

import java.util.function.Function;

/**
 * Enthält eine Auswahl verschiedener Aktivierungsfunktionen für neuronale
 * Netze. Jede Konstante stellt sowohl die Funktion selbst als auch ihre
 * Ableitung als {@link Function} bereit, sodass sie im Vorwärts- und im
 * Rückwärtsdurchlauf konsistent verwendet werden können.
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
    
    /**
     * Erstellt eine neue Aktivierungsfunktion aus Funktions- und Ableitungsreferenz.
     *
     * @param function   Die Aktivierungsfunktion.
     * @param derivative Ihre Ableitung.
     */
    
    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }
    
    /**
     * Wendet die Aktivierungsfunktion auf den übergebenen Wert an.
     *
     * @param x Der Eingabewert (gewichtete Summe inkl. Bias).
     * @return Der Aktivierungswert.
     */
    
    public double apply(double x) {
        return function.apply(x);
    }
    
    /**
     * Wertet die Ableitung der Aktivierungsfunktion an der Stelle {@code x} aus.
     * Wird im Backward-Pass zur Berechnung der Fehlersignale benötigt.
     *
     * @param x Der Eingabewert (gewichtete Summe inkl. Bias).
     * @return Der Wert der Ableitung an der Stelle {@code x}.
     */
    
    public double applyDerivative(double x) {
        return derivative.apply(x);
    }

    /**
     * Gibt die zugrundeliegende Aktivierungsfunktion als Referenz zurück.
     * Nützlich für Tests und zur Weitergabe an andere Komponenten.
     *
     * @return Die Funktionsreferenz.
     */
    public Function<Double, Double> getFunction() {
        return function;
    }
}
