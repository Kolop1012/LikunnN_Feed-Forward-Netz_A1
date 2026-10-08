import java.util.function.Function;

public class Layer {

    private Neuron[] neurons;
    private double[][] weights;

    /**
     * Erstellt die Eingabeschicht (Input Layer) des neuronalen Netzwerks.
     * <p>
     * Da die Neuronen dieser Schicht lediglich als Durchreicher für die externen
     * Eingabewerte dienen, besitzen sie standardmäßig keinen Bias-Wert (0.0)
     * und keine Gewichtsmatrix zu einer vorherigen Schicht.
     * </p>
     *
     * @param inputNodes Die Anzahl der Neuronen in dieser Eingabeschicht.
     */

    public Layer(int inputNodes, Function<Double, Double> activiationFunction) {
        neurons = new Neuron[inputNodes];
        for (int i = 0; i < inputNodes; i++) {
            neurons[i] = new Neuron(activiationFunction);
        }
    }

    /**
     * Erstellt eine verarbeitende Schicht (Hidden oder Output Layer) des Netzwerks.
     * <p>
     * Jedes Neuron dieser Schicht wird mit einem Bias von 0.0 initialisiert.
     * Zudem wird eine Gewichtsmatrix instanziiert, die die vollvermaschte
     * Verbindung
     * (Fully Connected) zur vorherigen Schicht darstellt. Die Gewichte werden
     * automatisch mit zufälligen Werten vorbelegt.
     * </p>
     *
     * @param currentNeurons  Die Anzahl der Neuronen in *dieser* Schicht.
     * @param previousNeurons Die Anzahl der Neuronen in der *vorherigen*
     *                        Schicht.
     */

    public Layer(int currentNeurons, int previousNeurons, Function<Double, Double> activiationFunction) {
        neurons = new Neuron[currentNeurons];
        for (int i = 0; i < currentNeurons; i++) {
            neurons[i] = new Neuron(activiationFunction);
        }

        this.weights = new double[currentNeurons][previousNeurons+1];
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = 1;  //änderen wenn nicht linear
            }
        }
    }

    /*
     * Getters
     */
    public Neuron[] getNeurons() {
        return neurons;
    }

    public double[][] getWeights() {
        return weights;
    }

}
