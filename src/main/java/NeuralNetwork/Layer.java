package NeuralNetwork;

import java.util.function.Function;

/**
 * Repräsentiert eine Schicht (Layer) innerhalb des neuronalen Netzwerks.
 * Eine Schicht besteht aus einem Array von Neuronen und, falls es sich nicht um
 * die
 * Eingangsschicht handelt, einer Gewichtsmatrix zur vorherigen Schicht.
 */

public class Layer {

    private final Neuron[] neurons;
    private final double[][] weights;

    /**
     * Erstellt die Eingabeschicht (Input Layer) des neuronalen Netzwerks.
     * <p>Diese Schicht besitzt keine <b>Gewichtsmatrix<b/>.
     *
     * @param inputNodes         Die Anzahl der Neuronen in dieser Eingabeschicht.
     * @param activationFunction Die Aktivierungsfunktion für die Neuronen.
     */
    public Layer(int inputNodes, Function<Double, Double> activationFunction) {
        neurons = new Neuron[inputNodes];
        for (int i = 0; i < inputNodes; i++) {
            neurons[i] = new Neuron(activationFunction);
        }
        this.weights = null;
    }

    /**
     * Erstellt eine verarbeitende Schicht (Hidden oder Output Layer) des Netzwerks.
     * Instanziiert zudem die vollvermaschte Gewichtsmatrix zur vorherigen Schicht.
     *
     * @param currentNeurons     Die Anzahl der Neuronen in *dieser* Schicht.
     * @param previousNeurons    Die Anzahl der Neuronen in der *vorherigen*
     *                           Schicht.
     * @param activationFunction Die Aktivierungsfunktion für die Neuronen.
     */
    public Layer(int currentNeurons, int previousNeurons, Function<Double, Double> activationFunction) {
        neurons = new Neuron[currentNeurons];
        for (int i = 0; i < currentNeurons; i++) {
            neurons[i] = new Neuron(activationFunction);
        }

        // Spalte für den Bias (previousNeurons + 1)
        this.weights = new double[currentNeurons][previousNeurons + 1];
        initWeights();
    }


    /**
     * Initialisiert die Gewichtsmatrix mit Startwerten (aktuell fix auf 1.0).
     *
     * <p><b>TODO:</b> Durch echte Zufallsinitialisierung (z.B. {@code Math.random()}) ersetzen.
     */
    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = 1.0;
            }
        }
    }

    /**
     * Führt einen Vorwärtsdurchlauf (Forward Pass) für diese Schicht durch.
     * Berechnet für jedes Neuron die gewichtete Summe der Eingaben aus der
     * Vorgängerschicht, addiert den Bias und wendet die Aktivierungsfunktion an.
     *
     * @param previousLayer Die vorherige Schicht im Netzwerk, deren Aktivierungen
     *                      als Eingabe dienen. Für die Eingangsschicht hat dieser
     *                      Parameter keine Auswirkungen.
     * @throws IllegalArgumentException wenn {@code previousLayer} die falsche Anzahl
     *                                  an Eingaben für die Gewichtsmatrix liefert.
     */
    public void forward(Layer previousLayer) {
        if (weights == null)
            return; // Input-Schicht hat keine Vorgänger-Gewichte

        double[] prevValues = previousLayer.getValues();

        for (int i = 0; i < neurons.length; i++) {
            double sum = 0;

            // Summiere (Wert des vorherigen Neurons * Gewicht)
            for (int j = 0; j < prevValues.length; j++) {
                sum += prevValues[j] * weights[i][j];
            }
            // Bias aufaddieren (letztes Element in der Gewichtszeile)
            sum += weights[i][prevValues.length];

            // Wende die Aktivierungsfunktion im Neuron an
            neurons[i].activate(sum);
        }
    }

    /**
     * Gibt alle Aktivierungswerte der Neuronen in dieser Schicht als Array zurück.
     *
     * @return Ein Array von Fließkommazahlen mit den aktuellen Aktivierungswerten.
     */
    public double[] getValues() {
        double[] values = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            values[i] = neurons[i].getValue();
        }
        return values;
    }

    /*
     * Getters
     */
    public Neuron[] getNeurons() {
        return neurons;
    }

}
