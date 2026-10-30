package NeuralNetwork.train;

import NeuralNetwork.model.CostFunction;
import NeuralNetwork.model.FFNeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Führt die Trainingsschleife aus und protokolliert den Loss-Verlauf.
 */
public class Trainer {
    
    private final FFNeuralNetwork net;
    private final double initialLearningRate;
    private final double momentum;
    private final LearningRateScheduler scheduler;
    
    public Trainer(FFNeuralNetwork net, String mode, double initialLearningRate) {
        this.net = net;
        this.initialLearningRate = initialLearningRate;
        this.momentum = mode.equals("sgd") ? 0.0 : 0.9;
        this.scheduler = mode.equals("adaptive") ? new LearningRateScheduler(200, 0.5, 0.01, 1e-5) : null;
    }
    
    public TrainResult train(double[][] inputs, double[][] targets, int epochs) {
        List<Double> history = new ArrayList<>();
        double lr = initialLearningRate;
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.feedForward(inputs[i]);
                net.train(lr, targets[i], momentum);
            }
            double loss = meanLoss(inputs, targets);
            history.add(loss);
            if (!Double.isFinite(loss)) {
                throw new IllegalArgumentException("Training instabil – Eingaben skalieren.");
            }
            if (epoch == 1 || epoch % 1000 == 0 || epoch == epochs) {
                System.out.printf(Locale.ROOT, "Epoche %5d | E %.6f | lr %.5f%n", epoch, loss, lr);
            }
            if (scheduler != null) {
                double next = scheduler.nextRate(lr, loss);
                if (next < lr)
                    System.out.printf(Locale.ROOT, "  lr %.5f -> %.5f%n", lr, next);
                lr = next;
            }
        }
        return new TrainResult(history, lr, history.getLast());
    }
    
    private double meanLoss(double[][] inputs, double[][] targets) {
        double total = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            total += CostFunction.HalfSquaredError.calculate(net.feedForward(inputs[i]), targets[i]);
        }
        return total / inputs.length;
    }
    
    public record TrainResult(List<Double> lossHistory, double finalLearningRate, double finalLoss) {
    }
}
