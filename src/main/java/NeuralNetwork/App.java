package NeuralNetwork;

import NeuralNetwork.config.AppConfig;
import NeuralNetwork.data.CsvParser;
import NeuralNetwork.data.Dataset;
import NeuralNetwork.data.Scaler;
import NeuralNetwork.eval.Evaluator;
import NeuralNetwork.model.ActivationFunction;
import NeuralNetwork.model.FFNeuralNetwork;
import NeuralNetwork.train.Trainer;
import NeuralNetwork.viz.NetworkVisualizer;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Einstiegspunkt. Delegiert Parsing an {@link AppConfig}, Training an
 * {@link Trainer} und Auswertung an {@link Evaluator}.
 */
public class App {
    
    static void main(String[] args) {
        // Muss vor dem ersten AWT-Zugriff gesetzt sein
        System.setProperty("java.awt.headless", "true");
        try {
            run(AppConfig.parse(args));
        } catch (IllegalArgumentException | IOException e) {
            System.err.println("Fehler: " + e.getMessage());
            System.err.println(
                    "Aufruf: NeuralNetwork.App [CSV] [sgd|momentum|adaptive] " + "[Epochen] [min-max] [hidden] [one-hot]");
            System.exit(1);
        }
    }
    
    /**
     * Kompatibilitäts-Wrapper für TestFFNeuralNetwork.
     */
    static double[][] readData(Path file) throws IOException {
        return CsvParser.parse(file);
    }
    
    private static void run(AppConfig cfg) throws IOException {
        Dataset ds = cfg.dataset();
        
        double[][] rawInputs = ds.extractAllInputs();
        double[][] rawTargets = ds.extractAllTargets();
        
        Scaler inputScaler = Scaler.fit(rawInputs);
        Scaler targetScaler = Scaler.fit(rawTargets);
        double[][] inputs = inputScaler.apply(rawInputs);
        double[][] targets = targetScaler.apply(rawTargets);
        
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ ds.inputCount(), cfg.hiddenNeurons(), ds.outputCount() },
                ActivationFunction.SIGMOID, 42L);
        
        System.out.printf("Muster: %d | Netz: %d-%d-%d | Modus: %s | Epochen: %d | One-Hot: %s%n", ds.size(),
                ds.inputCount(), cfg.hiddenNeurons(), ds.outputCount(), cfg.mode(), cfg.epochs(), cfg.oneHot());
        
        Trainer.TrainResult trainResult = new Trainer(net, cfg.mode(), 1.0).train(inputs, targets, cfg.epochs());
        
        Evaluator.EvalResult evalResult = new Evaluator(net, targetScaler, cfg.oneHot()).evaluate(inputs, rawTargets);
        
        try {
            NetworkVisualizer.saveVisualization("grafik_auswertung.png", trainResult.lossHistory(),
                    new int[]{ ds.inputCount(), cfg.hiddenNeurons(), ds.outputCount() }, net.getWeights(),
                    net.getBiases(), cfg.mode(), trainResult.finalLearningRate(), cfg.epochs(), evalResult.correct(),
                    evalResult.total());
        } catch (Throwable t) {
            System.err.println("Warnung: Visualisierung fehlgeschlagen (" + t.getClass()
                                                                             .getSimpleName() + ") – Training war erfolgreich.");
        }
    }
}
