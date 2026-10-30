package NeuralNetwork;

import java.util.Random;

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
    
    private final Layer[] layers;
    private boolean forwardReady;
    
    /**
     * Erstellt ein neues Feedforward-Netzwerk basierend auf der angegebenen
     * Topologie.
     *
     * @param topology       Ein Array, bei dem jedes Element die Anzahl der
     *                       Neuronen in
     *                       dieser Schicht angibt. Muss mindestens 2 Elemente
     *                       enthalten.
     * @param activationEnum Die Aktivierungsfunktion, die auf die
     *                       Neuronen angewendet wird.
     * @throws IllegalArgumentException Wenn die Topologie null ist oder weniger als
     *                                  2 Schichten enthält.
     */
    public FFNeuralNetwork(int[] topology, ActivationFunction activationEnum) {
        this(topology, activationEnum, new Random());
    }
    
    /**
     * Fester Seed für wiederholbare Versuche und Tests.
     *
     * @param topology       Schichtgrößen (mind. 2 Elemente).
     * @param activationEnum Aktivierungsfunktion für alle Neuronen.
     * @param seed           Seed für die Initialisierung der Gewichte.
     */
    
    public FFNeuralNetwork(int[] topology, ActivationFunction activationEnum, long seed) {
        this(topology, activationEnum, new Random(seed));
    }
    
    private FFNeuralNetwork(int[] topology, ActivationFunction activationEnum, Random random) {
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Das Netzwerk benötigt mindestens eine Input- und eine Output-Schicht.");
        }
        if (activationEnum == null) {
            throw new IllegalArgumentException("Eine Aktivierungsfunktion ist erforderlich.");
        }
        for (int size : topology) {
            if (size < 1) {
                throw new IllegalArgumentException("Jede Schicht benötigt mindestens ein Neuron.");
            }
        }
        layers = new Layer[topology.length];
        
        // Input-Schicht (hat keine Gewichte zur Vorgängerschicht)
        layers[0] = new Layer(topology[0], activationEnum);
        
        // Hidden- und Output-Schichten
        for (int i = 1; i < topology.length; i++) {
            layers[i] = new Layer(topology[i], topology[i - 1], activationEnum, random);
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
        forwardReady = false;
        if (inputValues == null || inputValues.length != layers[0].getNeurons().length) {
            throw new IllegalArgumentException("Anzahl der Eingabewerte stimmt nicht mit der Input-Schicht überein.");
        }
        for (double value : inputValues) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("Eingaben müssen endliche Zahlen sein.");
            }
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
        forwardReady = true;
        return layers[layers.length - 1].getValues();
    }
    
    /**
     * Ein Online-Lernschritt (Backpropagation + Gewichtsupdate) für das Ergebnis
     * des letzten Forward-Pass, ohne Momentum (entspricht {@code train(eta, targets, 0.0)}).
     *
     * @param learningRate Lernrate {@code eta > 0}.
     * @param targets      Zielwerte, ein Wert pro Ausgabeneuron.
     */
    
    public void train(double learningRate, double[] targets) {
        train(learningRate, targets, 0.0);
    }
    
    /**
     * Backpropagation für {@code E = 1/2 * Summe((Zielwert - Ausgabe)^2)},
     * optional mit Momentum. Erst werden alle Deltas berechnet, danach alle
     * Gewichte einschließlich Bias in einem Schritt aktualisiert.
     *
     * @param learningRate Lernrate {@code eta > 0}.
     * @param targets      Zielwerte, ein Wert pro Ausgabeneuron.
     * @param momentum     Momentum-Faktor im Bereich {@code [0, 1)}; {@code 0.0}
     *                     entspricht reinem Gradientenabstieg.
     * @throws IllegalArgumentException Bei ungültiger Lernrate, ungültigem
     *                                  Momentum oder falscher Zielwertanzahl.
     * @throws IllegalStateException    Wenn vor dem Aufruf kein
     *                                  {@link #feedForward(double[])} mit dem
     *                                  aktuellen Muster ausgeführt wurde.
     */
    public void train(double learningRate, double[] targets, double momentum) {
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException("Die Lernrate muss positiv und endlich sein.");
        }
        if (!Double.isFinite(momentum) || momentum < 0.0 || momentum >= 1.0) {
            throw new IllegalArgumentException("Momentum muss im Bereich [0, 1) liegen.");
        }
        int lastIndex = layers.length - 1;
        if (targets == null || targets.length != layers[lastIndex].getNeurons().length) {
            throw new IllegalArgumentException("Anzahl der Zielwerte stimmt nicht mit der Output-Schicht überein.");
        }
        for (double target : targets) {
            if (!Double.isFinite(target)) {
                throw new IllegalArgumentException("Zielwerte müssen endliche Zahlen sein.");
            }
        }
        if (!forwardReady) {
            throw new IllegalStateException("Vor jedem Lernschritt feedForward mit dem aktuellen Muster aufrufen.");
        }
        
        double[][] deltas = new double[layers.length][];
        deltas[lastIndex] = layers[lastIndex].calculateDeltas(null, null, targets);
        for (int i = lastIndex - 1; i > 0; i--) {
            deltas[i] = layers[i].calculateDeltas(deltas[i + 1], layers[i + 1].getWeights(), null);
        }
        for (int i = 1; i <= lastIndex; i++) {
            layers[i].updateWeights(deltas[i], learningRate, momentum);
        }
        forwardReady = false;
    }
    
    /**
     * Gibt die Schichten des Netzes zurück, beginnend mit der Eingabeschicht.
     * Wird in erster Linie für Tests und zur Inspektion der gelernten Gewichte
     * verwendet.
     *
     * @return Array aller {@link Layer}, Index 0 ist die Eingabeschicht.
     */
    public Layer[] getLayers() {
        return layers;
    }


    public void printWeightMatrices() {
        System.out.println("--- Gewichtsmatrizen ---");
        for (int l = 1; l < layers.length; l++) {
            System.out.println("Schicht " + l + " (Zeilen=aktuelle Neuronen, Spalten=Vorgänger + Bias):");
            double[][] w = layers[l].getWeights();
            for (int i = 0; i < w.length; i++) {
                System.out.print("Neuron " + i + ": [");
                for (int j = 0; j < w[i].length; j++) {
                    System.out.printf(java.util.Locale.ROOT, "%6.3f ", w[i][j]);
                }
                System.out.println("]");
            }
        }
    }
}
