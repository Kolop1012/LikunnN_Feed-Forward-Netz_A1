package NeuralNetwork.data;

import java.util.Arrays;

/**
 * Min-Max-Skalierung auf [0, 1] pro Spalte. Konstante Spalten werden auf 0.0 abgebildet.
 */
public class Scaler {
    
    private final double[] min;
    private final double[] max;
    
    private Scaler(double[] min, double[] max) {
        this.min = min;
        this.max = max;
    }
    
    public static Scaler fit(double[][] data) {
        int n = data[0].length;
        double[] min = new double[n];
        double[] max = new double[n];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);
        for (double[] row : data) {
            for (int k = 0; k < n; k++) {
                if (row[k] < min[k])
                    min[k] = row[k];
                if (row[k] > max[k])
                    max[k] = row[k];
            }
        }
        return new Scaler(min, max);
    }
    
    public double[][] apply(double[][] data) {
        double[][] result = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            result[i] = new double[data[i].length];
            for (int k = 0; k < data[i].length; k++) {
                result[i][k] = (min[k] == max[k]) ? 0.0 : (data[i][k] - min[k]) / (max[k] - min[k]);
            }
        }
        return result;
    }
    
    public double unscale(int idx, double value) {
        return (min[idx] == max[idx]) ? min[idx] : value * (max[idx] - min[idx]) + min[idx];
    }
}
