import java.util.List;
import java.util.ArrayList;

/**
 * Repräsentiert ein mehrschichtiges Feedforward Neuronales Netzwerk (Multilayer
 * Perceptron).
 * <p>
 * Die Daten fließen in diesem Netzwerk ausschließlich vorwärts
 * (vorwärtsgerichtet)
 * von der Eingabeschicht über die verborgenen Schichten bis zur Ausgabeschicht.
 * </p>
 */

public class FFNeuralNetwork {

    private final List<Layer> layers = new ArrayList<>();

    /**
     * Erstellt ein neues Feedforward-Netzwerk basierend auf der angegebenen
     * Topologie.
     * <p>
     * Beispiel: {@code new FFNeuralNetwork(new int[]{3, 4, 2})} erzeugt ein
     * Netzwerk mit
     * 3 Eingängen, einer verdeckten Schicht (Hidden Layer) mit 4 Neuronen und 2
     * Ausgängen.
     * </p>
     *
     * @param topology Ein Array, bei dem jedes Element die Anzahl der Neuronen in
     *                 dieser Schicht angibt.
     *                 Muss mindestens 2 Elemente enthalten (Input und Output).
     * @throws IllegalArgumentException Wenn die Topologie null ist oder weniger als
     *                                  2 Schichten enthält.
     */

    public FFNeuralNetwork(int[] topology) {

        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Das Netzwerk benötigt mindestens eine Input- und eine Output-Schicht.");
        }
        layers.add(new Layer(topology[0]));
        for (int i = 1; i < topology.length; i++) {
            layers.add(new Layer(topology[i], topology[i - 1]));
        }
    }

    // Aktivierungsfunktion
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Führt den Vorwärtsfluss (Feedforward) der Daten durch das Netzwerk aus.
     * Die Eingabewerte werden durch alle Schichten transportiert, mit den Gewichten
     * multipliziert, mit dem Bias summiert und durch die Aktivierungsfunktion
     * transformiert.
     *
     * @param inputValues Die externen Eingabewerte für das Netzwerk.
     * @return Ein Array mit den berechneten Aktivierungswerten der Ausgabeschicht.
     * @throws IllegalArgumentException Wenn die Anzahl der Eingabewerte nicht exakt
     *                                  mit der
     *                                  Größe der Input-Schicht übereinstimmt.
     */

    public double[] feedForward(double[] inputValues) {

        // Setze den Input Layer
        Layer inputLayer = layers.get(0);
        if (inputValues.length != inputLayer.getNeurons().size()) {
            throw new IllegalArgumentException("Anzahl der Eingabewerte stimmt nicht mit der Input-Schicht überein.");
        }

        for (int i = 0; i < inputValues.length; i++) {
            inputLayer.getNeurons().get(i).setValue(inputValues[i]);
        }

        // Berechnet alle Layer
        for (int l = 1; l < layers.size(); l++) {
            Layer currentLayer = layers.get(l);
            Layer previousLayer = layers.get(l - 1);
            double[][] weights = currentLayer.getWeights();

            for (int i = 0; i < currentLayer.getNeurons().size(); i++) {
                double sum = currentLayer.getNeurons().get(i).getBias();

                // Summiere (Wert des vorherigen Neurons * Gewicht)
                for (int j = 0; j < previousLayer.getNeurons().size(); j++) {
                    sum += previousLayer.getNeurons().get(j).getValue() * weights[i][j];
                }
                currentLayer.getNeurons().get(i).setValue(sigmoid(sum));
            }
        }

        // Output ausgeben
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.getNeurons().size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeurons().get(i).getValue();
        }
        return output;
    }
}
