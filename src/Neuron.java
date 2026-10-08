import java.util.function.Function;

/**
 * Repräsentiert ein einzelnes Neuron (Knoten) innerhalb einer Schicht des
 * Netzwerks.
 * Speichert den aktuellen Aktivierungswert sowie den Schwellenwert (Bias).
 */

public class Neuron {

    private double value;
    private final Function<Double, Double> activiationFunction;

    /**
     * Erstellt ein neues Neuron mit dem angegebenen Schwellenwert.
     * Der Aktivierungswert wird initial auf 0.0 gesetzt.
     *
     * @param activiationFunction Der anfängliche Schwellenwert (Bias) des Neurons.
     */

    public Neuron(Function<Double, Double> activiationFunction) {
        this.activiationFunction = activiationFunction;
        this.value = 0.0;
    }

    public void caluculate() {
        value = activiationFunction.apply(value);
    }
    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

}
