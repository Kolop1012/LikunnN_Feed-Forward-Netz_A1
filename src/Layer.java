import java.util.List;
import java.util.ArrayList;

public class Layer {

    private final List<Neuron> neurons = new ArrayList<>();
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

    public Layer(int inputNodes) {
        for (int i = 0; i < inputNodes; i++) {
            neurons.add(new Neuron(0.0));
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

    public Layer(int currentNeurons, int previousNeurons) {
        for (int i = 0; i < currentNeurons; i++) {
            neurons.add(new Neuron(0.0));
        }

        this.weights = new double[currentNeurons][previousNeurons];
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (Math.random() - 0.5);
            }
        }
    }

    /*
     * Getters
     */
    public List<Neuron> getNeurons() {
        return neurons;
    }

    public double[][] getWeights() {
        return weights;
    }

}
