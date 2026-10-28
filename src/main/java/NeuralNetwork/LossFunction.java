package NeuralNetwork;

/**
 * Ein funktionales Interface zur Definition von Kosten- oder Verlustfunktionen
 * (Loss Functions) für neuronale Netze.
 */
@FunctionalInterface
public interface LossFunction {
    
    /**
     * Berechnet den Fehlerwert (Loss) zwischen den berechneten Ausgaben
     * und den erwarteten Zielwerten.
     *
     * @param outputs Berechneter Output des Feed-Forward-Netzwerks.
     * @param targets Erwarteter Zielwert des Feed-Forward-Netzwerks.
     * @return Den berechneten Fehler als {@code double}.
     */
    double calculate(double[] outputs, double[] targets);
}
