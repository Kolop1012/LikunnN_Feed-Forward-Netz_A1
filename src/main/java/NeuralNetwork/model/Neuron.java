package NeuralNetwork.model;

/**
 * Repräsentiert ein einzelnes Neuron (Knoten) innerhalb einer Schicht des
 * Netzwerks. Speichert den aktuellen Aktivierungswert sowie die Referenz auf
 * die zugehörige Aktivierungsfunktion.
 */
public class Neuron {
    
    private final ActivationFunction activationFunction;
    private double value;
    
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
    
    /**
     * Gibt den aktuellen Aktivierungswert des Neurons zurück.
     *
     * @return Der aktuelle Wert.
     */
    public double getValue() {
        return value;
    }
    
    /**
     * Setzt den Aktivierungswert direkt. Wird für die Eingabeschicht verwendet,
     * um die externen Eingabewerte einzutragen, ohne die Aktivierungsfunktion
     * anzuwenden.
     *
     * @param value Der neue Aktivierungswert.
     */
    public void setValue(double value) {
        this.value = value;
    }
    
    /**
     * Gibt die Aktivierungsfunktion dieses Neurons zurück.
     * Wird im Backward-Pass benötigt, um die Ableitung auszuwerten.
     *
     * @return Die Aktivierungsfunktion des Neurons.
     */
    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }
}
