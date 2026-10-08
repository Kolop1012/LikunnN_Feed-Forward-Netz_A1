package NeuralNetwork;

/**
 * Repräsentiert ein mehrschichtiges Feedforward Neuronales Netzwerk (Multilayer
 * Perceptron).
 * <p>
 * Die Daten fließen in diesem Netzwerk ausschließlich vorwärts
 * von der Eingabeschicht über die verborgenen Schichten (Hidden Layers) bis zur
 * Ausgabeschicht.
 * </p>
 */
public class FFNeuralNetwork {

    private Layer[] layers;

    /**
     * Erstellt ein neues Feedforward-Netzwerk basierend auf der angegebenen
     * Topologie.
     *
     * @param topology           Ein Array, bei dem jedes Element die Anzahl der
     *                           Neuronen in
     *                           dieser Schicht angibt. Muss mindestens 2 Elemente
     *                           enthalten.
     * @param activationFunction Die globale Aktivierungsfunktion, die auf die
     *                           Neuronen angewendet wird.
     * @throws IllegalArgumentException Wenn die Topologie null ist oder weniger als
     *                                  2 Schichten enthält.
     */
    public FFNeuralNetwork(int[] topology, ActivationFunction activationEnum) {
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Das Netzwerk benötigt mindestens eine Input- und eine Output-Schicht.");
        }
        layers = new Layer[topology.length];

        // Input-Schicht (hat keine Gewichte zur Vorsicht)
        layers[0] = new Layer(topology[0], activationEnum.getFunction());

        // Hidden- und Output-Schichten
        for (int i = 1; i < topology.length; i++) {
            layers[i] = new Layer(topology[i], topology[i - 1], activationEnum.getFunction());
        }
    }

    /**
     * Führt den Vorwärtsfluss (Feedforward) der Daten durch das gesamte Netzwerk
     * aus.
     * Die Eingabewerte werden in die Eingabeschicht geschrieben und schichtweise
     * durch das Netz propagiert.
     *
     * @param inputValues Die externen Eingabewerte für das Netzwerk.
     * @return Ein Array mit den berechneten Aktivierungswerten der Ausgabeschicht.
     * @throws IllegalArgumentException Wenn die Anzahl der Eingabewerte nicht exakt
     *                                  mit der Größe der Input-Schicht
     *                                  übereinstimmt.
     */
    public double[] feedForward(double[] inputValues) {
        if (inputValues.length != layers[0].getNeurons().length) {
            throw new IllegalArgumentException("Anzahl der Eingabewerte stimmt nicht mit der Input-Schicht überein.");
        }

        // 1. Eingabewerte in die Input-Schicht schreiben
        Neuron[] inputNeurons = layers[0].getNeurons();
        for (int i = 0; i < inputValues.length; i++) {
            inputNeurons[i].setValue(inputValues[i]);
        }

        // 2. Schicht für Schicht nach vorne durchreichen (berechnet Gewichte &
        // Aktivierung)
        for (int l = 1; l < layers.length; l++) {
            layers[l].forward(layers[l - 1]);
        }

        // 3. Ausgaben der letzten Schicht zurückgeben
        return layers[layers.length - 1].getValues();
    }
}
