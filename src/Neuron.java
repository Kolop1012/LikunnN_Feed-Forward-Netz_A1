/**
 * Repräsentiert ein einzelnes Neuron (Knoten) innerhalb einer Schicht des
 * Netzwerks.
 * Speichert den aktuellen Aktivierungswert sowie den Schwellenwert (Bias).
 */

public class Neuron {

    private double value;
    private double bias;

    /**
     * Erstellt ein neues Neuron mit dem angegebenen Schwellenwert.
     * Der Aktivierungswert wird initial auf 0.0 gesetzt.
     *
     * @param bias Der anfängliche Schwellenwert (Bias) des Neurons.
     */

    public Neuron(double bias) {
        this.bias = bias;
        this.value = 0.0;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getBias() {
        return bias;
    }

}
