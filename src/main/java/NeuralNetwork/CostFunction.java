package NeuralNetwork;

/**
 * Repräsentiert verschiedene Kostenfunktionen (Loss Functions) zur Bewertung
 * der Abweichung zwischen den vorhergesagten Ausgaben und den Zielwerten
 * in einem neuronalen Netzwerk.
 * <p>
 * Implementiert das {@link LossFunction}-Interface, wodurch die Enum-Konstanten
 * direkt als ausführbare Funktionen genutzt werden können.
 * </p>
 */

public enum CostFunction implements LossFunction {

    /** Trainingsfehler passend zur Delta-Regel der Folien: halbe Fehlerquadratsumme. */
    HalfSquaredError((outputs, targets) -> {
        double sum = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            double error = targets[i] - outputs[i];
            sum += error * error;
        }
        return 0.5 * sum;
    }),

    /**
     * Mean Squared Error (MSE).
     * Berechnet den durchschnittlichen quadrierten Abstand zwischen den
     * berechneten Ausgaben und den Zielwerten.
     */


    MeanSquaredError((outputs, targets) -> {
        int length = outputs.length;
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            double error = outputs[i] - targets[i];
            sum += error * error;
        }
        return sum / outputs.length;
    }
    );


    private final LossFunction lossFunction;

    /**
     * Erstellt eine neue Kostenfunktion basierend auf dem übergebenen Lambda-Ausdruck.
     *
     * @param lossFunc Das funktionale Interface, das die Berechnungslogik definiert.
     */

    CostFunction(LossFunction lossFunc) {
        this.lossFunction = lossFunc;
    }

    /**
     * Berechnet den Fehlerwert (Loss) anhand der Netzwerkausgaben und der Zielwerte.
     *
     * @param outputs Die vom neuronalen Netzwerk berechneten Ausgabewerte.
     * @param targets Die erwarteten Zielwerte (Ground Truth).
     * @return Den berechneten Fehler als {@code double}.
     */

    @Override
    public double calculate(double[] outputs, double[] targets) {
        if (outputs == null || targets == null || outputs.length == 0 || outputs.length != targets.length) {
            throw new IllegalArgumentException("Ausgaben und Zielwerte müssen gleich lang und nicht leer sein.");
        }
        return lossFunction.calculate(outputs, targets);
    }


}
