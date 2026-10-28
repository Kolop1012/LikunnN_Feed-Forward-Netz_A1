package NeuralNetwork;

import java.util.Random;

/**
 * Repräsentiert eine Schicht (Layer) innerhalb des neuronalen Netzwerks.
 * Eine Schicht besteht aus einem Array von Neuronen und, falls es sich nicht um
 * die
 * Eingangsschicht handelt, einer Gewichtsmatrix zur vorherigen Schicht.
 */

public class Layer {
    
    private final Neuron[] neurons;
    private double[][] weights;
    private double[] lastInputs;
    private double[] lastRawInputs;
    private double[][] previousUpdates;
    
    /**
     * Erstellt die Eingabeschicht (Input Layer) des neuronalen Netzwerks.
     * Diese Schicht besitzt keine Gewichtsmatrix.
     *
     * @param inputNodes     Die Anzahl der Neuronen in dieser Eingabeschicht.
     * @param activationEnum Die Aktivierungsfunktion für die Neuronen.
     */
    public Layer(int inputNodes, ActivationFunction activationEnum) {
        neurons = new Neuron[inputNodes];
        for (int i = 0; i < inputNodes; i++) {
            neurons[i] = new Neuron(activationEnum);
        }
        this.weights = null;
    }
    
    /**
     * Erstellt eine verarbeitende Schicht (Hidden oder Output Layer) des Netzwerks.
     * Instanziiert zudem die vollvermaschte Gewichtsmatrix zur vorherigen Schicht.
     *
     * @param currentNeurons  Die Anzahl der Neuronen in dieser Schicht.
     * @param previousNeurons Die Anzahl der Neuronen in der vorherigen Schicht.
     * @param activationEnum  Die Aktivierungsfunktion für die Neuronen.
     */
    public Layer(int currentNeurons, int previousNeurons, ActivationFunction activationEnum) {
        this(currentNeurons, previousNeurons, activationEnum, new Random());
    }
    
    /**
     * Interner Konstruktor mit injizierbarem Zufallsgenerator (für Tests mit festem Seed).
     *
     * @param currentNeurons  Die Anzahl der Neuronen in dieser Schicht.
     * @param previousNeurons Die Anzahl der Neuronen in der vorherigen Schicht.
     * @param activationEnum  Die Aktivierungsfunktion für die Neuronen.
     * @param random          Der Zufallsgenerator für die Initialisierung.
     */
    Layer(int currentNeurons, int previousNeurons, ActivationFunction activationEnum, Random random) {
        neurons = new Neuron[currentNeurons];
        for (int i = 0; i < currentNeurons; i++) {
            neurons[i] = new Neuron(activationEnum);
        }
        
        // Spalte für den Bias (previousNeurons + 1)
        this.weights = new double[currentNeurons][previousNeurons + 1];
        this.previousUpdates = new double[currentNeurons][previousNeurons + 1];
        initWeights(random, previousNeurons);
    }
    
    /**
     * Initialisiert die Gewichte mit symmetrischen Zufallswerten (Xavier-Bereich).
     * Die Bias-Werte starten bei null.
     *
     * @param random          Der Zufallsgenerator.
     * @param previousNeurons Die Anzahl der Neuronen in der vorherigen Schicht.
     */
    private void initWeights(Random random, int previousNeurons) {
        double limit = Math.sqrt(6.0 / (previousNeurons + neurons.length));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < previousNeurons; j++) {
                weights[i][j] = (random.nextDouble() * 2.0 - 1.0) * limit;
            }
        }
    }
    
    /**
     * Führt einen Vorwärtsdurchlauf (Forward Pass) für diese Schicht durch.
     * Berechnet für jedes Neuron die gewichtete Summe der Eingaben aus der
     * Vorgängerschicht, addiert den Bias und wendet die Aktivierungsfunktion an.
     * Die Zwischenwerte werden für den späteren Backward-Pass gespeichert.
     *
     * @param previousLayer Die vorherige Schicht im Netzwerk, deren Aktivierungen
     *                      als Eingabe dienen. Für die Eingangsschicht hat dieser
     *                      Parameter keine Auswirkungen.
     * @throws IllegalArgumentException Wenn {@code previousLayer} die falsche Anzahl
     *                                  an Eingaben für die Gewichtsmatrix liefert.
     */
    public void forward(Layer previousLayer) {
        if (weights == null)
            return; // Input-Schicht hat keine Vorgänger-Gewichte
        
        double[] prevValues = previousLayer.getValues();
        if (prevValues.length != weights[0].length - 1) {
            throw new IllegalArgumentException("Die Vorgängerschicht passt nicht zur Gewichtsmatrix.");
        }
        this.lastInputs = prevValues;
        this.lastRawInputs = new double[neurons.length];
        
        for (int i = 0; i < neurons.length; i++) {
            double sum = 0;
            
            // Summiere (Wert des vorherigen Neurons * Gewicht)
            for (int j = 0; j < prevValues.length; j++) {
                sum += prevValues[j] * weights[i][j];
            }
            // Bias aufaddieren (letztes Element in der Gewichtszeile)
            sum += weights[i][prevValues.length];
            
            lastRawInputs[i] = sum;
            // Wende die Aktivierungsfunktion im Neuron an
            neurons[i].activate(sum);
        }
    }
    
    /**
     * Gibt alle Aktivierungswerte der Neuronen in dieser Schicht als Array zurück.
     *
     * @return Ein Array mit den aktuellen Aktivierungswerten in Neuronenreihenfolge.
     */
    public double[] getValues() {
        double[] values = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            values[i] = neurons[i].getValue();
        }
        return values;
    }
    
    /**
     * Gibt die Neuronen dieser Schicht zurück.
     *
     * @return Das Array der Neuronen.
     */
    public Neuron[] getNeurons() {
        return neurons;
    }
    
    /**
     * Gibt die Gewichtsmatrix dieser Schicht zurück. Zeile {@code i} enthält die
     * eingehenden Gewichte von Neuron {@code i}; die letzte Spalte ist der Bias.
     * Für die Eingabeschicht ist der Rückgabewert {@code null}.
     *
     * @return Die Gewichtsmatrix oder {@code null} für die Eingabeschicht.
     */
    public double[][] getWeights() {
        return weights;
    }
    
    /**
     * Berechnet die Fehlersignale (Deltas) dieser Schicht und verändert noch keine
     * Gewichte. Für die Ausgabeschicht werden die Zielwerte direkt verglichen;
     * für verborgene Schichten wird der Fehler über die Gewichte der Nachfolgerschicht
     * zurückgeführt.
     *
     * @param nextLayerDeltas  Deltas der Nachfolgerschicht (nur für verborgene Schichten).
     * @param nextLayerWeights Gewichte der Nachfolgerschicht (nur für verborgene Schichten).
     * @param targets          Zielwerte (nur für die Ausgabeschicht; sonst {@code null}).
     * @return Die berechneten Deltas, ein Wert pro Neuron dieser Schicht.
     */
    double[] calculateDeltas(double[] nextLayerDeltas, double[][] nextLayerWeights, double[] targets) {
        int currentSize = neurons.length;
        double[] currentDeltas = new double[currentSize];
        
        // 1. Deltas für diese Schicht berechnen
        if (targets != null) {
            // Fall A: Ausgabeschicht (Output Layer)
            for (int i = 0; i < currentSize; i++) {
                double output = neurons[i].getValue();
                double derivative = neurons[i].getActivationFunction().applyDerivative(lastRawInputs[i]);
                // Folie 10: Delta = f'(net) * (Zielwert - Ausgabe).
                currentDeltas[i] = (targets[i] - output) * derivative;
            }
        } else {
            // Fall B: Versteckte Schicht (Hidden Layer) - Fehler von rechts aufteilen
            for (int j = 0; j < currentSize; j++) {
                double sumError = 0;
                for (int k = 0; k < nextLayerWeights.length; k++) {
                    // Multiplikation mit transponierten Gewichten der rechten Schicht
                    sumError += nextLayerDeltas[k] * nextLayerWeights[k][j];
                }
                double derivative = neurons[j].getActivationFunction().applyDerivative(lastRawInputs[j]);
                currentDeltas[j] = sumError * derivative;
            }
        }
        
        return currentDeltas;
    }
    
    /**
     * Aktualisiert alle Gewichte und Bias-Werte dieser Schicht gemäß der Delta-Regel,
     * optional mit Momentum. Die letzte Spalte der Gewichtsmatrix enthält den Bias
     * (Eingang konstant 1.0); die letzte Änderung wird für den nächsten Schritt
     * in {@code previousUpdates} gespeichert.
     *
     * @param deltas       Die zuvor berechneten Deltas dieser Schicht.
     * @param learningRate Lernrate {@code eta > 0}.
     * @param momentum     Momentum-Faktor im Bereich {@code [0, 1)}.
     */
    void updateWeights(double[] deltas, double learningRate, double momentum) {
        int prevSize = lastInputs.length;
        for (int i = 0; i < neurons.length; i++) {
            for (int j = 0; j <= prevSize; j++) {
                double input = j == prevSize ? 1.0 : lastInputs[j];
                double update = momentum * previousUpdates[i][j] + learningRate * deltas[i] * input;
                weights[i][j] += update;
                previousUpdates[i][j] = update;
            }
        }
    }
}
