package NeuralNetwork.data;

/**
 * Kapselt die Rohdaten und liefert Feature-/Target-Extraktion.
 */
public class Dataset {
    
    private final double[][] rawData;
    private final int[] outputRange;
    
    public Dataset(double[][] rawData, int[] outputRange) {
        this.rawData = rawData;
        this.outputRange = outputRange;
    }
    
    public int size() {
        return rawData.length;
    }
    
    public int inputCount() {
        return rawData[0].length - outputCount();
    }
    
    public int outputCount() {
        return outputRange[1] - outputRange[0] + 1;
    }
    
    public double[] extractInput(int row) {
        double[] inputs = new double[inputCount()];
        int idx = 0;
        for (int i = 0; i < rawData[row].length; i++) {
            if (i < outputRange[0] || i > outputRange[1])
                inputs[idx++] = rawData[row][i];
        }
        return inputs;
    }
    
    public double[] extractTarget(int row) {
        double[] targets = new double[outputCount()];
        for (int i = 0; i < outputCount(); i++)
            targets[i] = rawData[row][outputRange[0] + i];
        return targets;
    }
    
    public double[][] extractAllInputs() {
        double[][] result = new double[size()][];
        for (int i = 0; i < size(); i++)
            result[i] = extractInput(i);
        return result;
    }
    
    public double[][] extractAllTargets() {
        double[][] result = new double[size()][];
        for (int i = 0; i < size(); i++)
            result[i] = extractTarget(i);
        return result;
    }
}
