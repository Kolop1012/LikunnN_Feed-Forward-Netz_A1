package NeuralNetwork.eval;

import NeuralNetwork.data.Scaler;
import NeuralNetwork.model.FFNeuralNetwork;

import java.util.Locale;

/**
 * Wertet das trainierte Netz aus - Argmax bei One-Hot, Rundung sonst.
 */
public class Evaluator {
    
    private final FFNeuralNetwork net;
    private final Scaler targetScaler;
    private final boolean oneHot;
    
    public Evaluator(FFNeuralNetwork net, Scaler targetScaler, boolean oneHot) {
        this.net = net;
        this.targetScaler = targetScaler;
        this.oneHot = oneHot;
    }
    
    public EvalResult evaluate(double[][] inputs, double[][] rawTargets) {
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] out = net.feedForward(inputs[i]);
            double[] finalOut = new double[out.length];
            for (int k = 0; k < out.length; k++)
                finalOut[k] = targetScaler.unscale(k, out[k]);
            
            int predicted = oneHot ? argmax(finalOut) : (int) Math.round(finalOut[0]);
            int actual = oneHot ? argmax(rawTargets[i]) : (int) Math.round(rawTargets[i][0]);
            if (predicted == actual)
                correct++;
        }
        double rate = 100.0 * correct / inputs.length;
        System.out.printf(Locale.ROOT, "Korrekt: %d/%d (%.1f%%)%n", correct, inputs.length, rate);
        return new EvalResult(correct, inputs.length, rate);
    }
    
    private static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }
    
    public record EvalResult(int correct, int total, double accuracy) {
    }
}
