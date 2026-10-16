package NeuralNetwork;

/**
 * Repräsentiert ein einzelnes Neuron (Knoten) innerhalb einer Schicht des
 * Netzwerks.
 * Speichert den aktuellen Aktivierungswert sowie die Referenz auf die
 * zugehörige Aktivierungsfunktion.
 */
public class Neuron {

    private double value;
    private final ActivationFunction activationFunction;

    /**
     * Erstellt ein neues Neuron mit der übergebenen Aktivierungsfunktion.
     * Der initiale Aktivierungswert wird auf 0.0 gesetzt.
     *
     * @param activationEnum Die Aktivierungsfunktion für dieses Neuron.
     */
    public Neuron(ActivationFunction activationEnum) {
        this.activationFunction = activationEnum;
        this.value = 0.0;
    }

    /**
     * Berechnet den neuen Aktivierungswert, indem die Aktivierungsfunktion
     * auf den gewichteten rohen Eingangswert angewendet wird.
     *
     * @param rawInput Der gewichtete Summenwert inklusive Bias.
     */
    public void activate(double rawInput) {
        this.value = activationFunction.apply(rawInput);
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public ActivationFunction getActivationFunction () {
        return this.activationFunction;
    }
}
