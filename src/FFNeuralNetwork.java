import java.util.function.Function;

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

    private Layer[] layers;

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

    public FFNeuralNetwork(int[] topology, Function<Double, Double> activiationFunction) {
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Das Netzwerk benötigt mindestens eine Input- und eine Output-Schicht.");
        }
        layers = new Layer[topology.length];
        layers[0] = new Layer(topology[0], activiationFunction);
        for (int i = 1; i < topology.length; i++) {
            layers[i] = new Layer(topology[i], topology[i - 1], activiationFunction);
        }
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
        if (inputValues.length != layers[0].getNeurons().length) {
            throw new IllegalArgumentException("Anzahl der Eingabewerte stimmt nicht mit der Input-Schicht überein.");
        }

        for (int i = 0; i < inputValues.length; i++) {
            layers[0].getNeurons()[i].setValue(inputValues[i]);
        }

        // Berechnet alle Layer
        for (int l = 1; l < layers.length; l++) {
            Layer currentLayer = layers[l];
            Layer previousLayer = layers[l-1];
            double[][] weights = currentLayer.getWeights();

            for (int i = 0; i < currentLayer.getNeurons().length; i++) {
                double sum = 0;

                // Summiere (Wert des vorherigen Neurons * Gewicht)
                for (int j = 0; j < previousLayer.getNeurons().length; j++) {
                    sum += previousLayer.getNeurons()[j].getValue() * weights[i][j];
                }
                sum += weights[i][previousLayer.getNeurons().length];
                currentLayer.getNeurons()[i].setValue(sum);
            }
        }

        // Output ausgeben
        Layer outputLayer = layers[layers.length-1];
        double[] output = new double[outputLayer.getNeurons().length];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeurons()[i].getValue();
        }
        return output;
    }
}
