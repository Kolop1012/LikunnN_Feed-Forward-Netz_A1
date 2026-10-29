package NeuralNetwork;

import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
/**
 * Ausführbare Tests ohne zusätzliche Bibliotheken: java -cp bin NeuralNetwork.TestFFNeuralNetwork.
 */
public class TestFFNeuralNetwork {
    private static int passed;
    
    public static void main(String[] args) throws Exception {
        testForwardPass();
        testCostFunction();
        testGradients(ActivationFunction.LINEAR, new int[]{ 2, 2 });
        testGradients(ActivationFunction.SIGMOID, new int[]{ 2, 3, 2, 3 });
        testGradients(ActivationFunction.TANH, new int[]{ 2, 3, 1 });
        testGradients(ActivationFunction.RELU, new int[]{ 2, 3, 1 });
        testMomentum();
        testScheduler();
        testXor(0.0, false);
        testXor(0.9, false);
        testXor(0.9, true);
        testValidation();
        testCsv();
        System.out.println("Alle " + passed + " Testgruppen bestanden.");
    }
    
    private static void testForwardPass() throws Exception {
        int[][] topologies = { { 1, 1 }, { 3, 1, 1, 1 }, { 5, 3, 2, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 } };
        double[][] inputs = { { 1 }, { 1, 1, 1 }, { -1, 0, 1, 2, -3 }, { 1 } };
        double[] expected = { 2, 6, 3, 9864101 };
        for (int t = 0; t < topologies.length; t++) {
            FFNeuralNetwork net = new FFNeuralNetwork(topologies[t], ActivationFunction.LINEAR, 42L);
            Layer[] layers = layersOf(net);
            for (int l = 1; l < layers.length; l++) {
                for (double[] row : layers[l].getWeights()) {
                    java.util.Arrays.fill(row, 1.0);
                }
            }
            for (double output : net.feedForward(inputs[t])) {
                near(expected[t], output, 1e-9, "Forward-Pass mit festen Gewichten");
            }
        }
        pass("Forward-Pass: vier Topologien mit definierten Gewichten");
    }
    
    private static void testCostFunction() {
        near(0.5, CostFunction.HalfSquaredError.calculate(new double[]{ 1 }, new double[]{ 0 }), 0.0, "Halbe Fehlerquadratsumme bei einer Ausgabe");
        double[] output = { 3, 4, 5 };
        double[] target = { 1, 1, 1 };
        near(14.5, CostFunction.HalfSquaredError.calculate(output, target), 0.0, "Trainingsfehler bei drei Ausgaben ohne Mittelung");
        near(29.0 / 3.0, CostFunction.MeanSquaredError.calculate(output, target), 1e-12, "MSE bleibt als eigenständige Messgröße verfügbar");
        pass("Fehlerfunktion der Folien: halbe Fehlerquadratsumme");
    }
    
    /**
     * Vergleicht die Updates mit zentralen finiten Differenzen der halben Fehlerquadratsumme.
     */
    private static void testGradients(ActivationFunction activation, int[] topology) throws Exception {
        FFNeuralNetwork net = new FFNeuralNetwork(topology, activation, 42L);
        Layer[] layers = layersOf(net);
        double[] input = { 0.3, 0.7 };
        int outputs = topology[topology.length - 1];
        double[] target = outputs == 3 ? new double[]{ 0.2, 0.8, 0.4 } : outputs == 2 ? new double[]{ 0.2, 0.8 } : new double[]{ 0.2 };
        if (activation == ActivationFunction.RELU) {
            // Positive Summen, damit der Test nicht an der nicht differenzierbaren Null liegt.
            for (int l = 1; l < layers.length; l++) {
                for (double[] row : layers[l].getWeights()) {
                    java.util.Arrays.fill(row, 0.2);
                }
            }
        }
        double[][][] before = new double[layers.length][][];
        double[][][] gradients = new double[layers.length][][];
        double epsilon = 1e-6;
        for (int l = 1; l < layers.length; l++) {
            double[][] weights = layers[l].getWeights();
            before[l] = new double[weights.length][];
            gradients[l] = new double[weights.length][];
            for (int i = 0; i < weights.length; i++) {
                before[l][i] = weights[i].clone();
                gradients[l][i] = new double[weights[i].length];
                for (int j = 0; j < weights[i].length; j++) {
                    double original = weights[i][j];
                    weights[i][j] = original + epsilon;
                    double plus = loss(net, input, target);
                    weights[i][j] = original - epsilon;
                    double minus = loss(net, input, target);
                    weights[i][j] = original;
                    gradients[l][i][j] = (plus - minus) / (2.0 * epsilon);
                }
            }
        }
        double rate = 0.05;
        net.feedForward(input);
        net.train(rate, target);
        for (int l = 1; l < layers.length; l++) {
            double[][] weights = layers[l].getWeights();
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    double actualGradient = (before[l][i][j] - weights[i][j]) / rate;
                    near(gradients[l][i][j], actualGradient, 1e-7, "Gradient " + activation + " Schicht " + l);
                }
            }
        }
        pass("Numerische Gradienten inklusive Bias: " + activation);
    }
    
    private static void testMomentum() throws Exception {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 1, 1 }, ActivationFunction.LINEAR, 42L);
        double[][] weights = layersOf(net)[1].getWeights();
        weights[0][0] = 0.5;
        weights[0][1] = 0.1;
        net.feedForward(new double[]{ 2.0 });
        net.train(0.1, new double[]{ 0.0 }, 0.9);
        near(0.28, weights[0][0], 1e-12, "Erstes Gewichtsupdate nach der Delta-Regel");
        near(-0.01, weights[0][1], 1e-12, "Erstes Bias update nach der Delta-Regel");
        // Zielwert gleich Ausgabe: Delta ist null, nur das Momentum wirkt weiter.
        double[] output = net.feedForward(new double[]{ 2.0 });
        net.train(0.1, output, 0.9);
        near(0.082, weights[0][0], 1e-12, "Momentum des Gewichts");
        near(-0.109, weights[0][1], 1e-12, "Momentum des Bias");
        pass("Momentum über zwei Lernschritte einschließlich Bias");
    }
    
    private static void testScheduler() {
        LearningRateScheduler scheduler = new LearningRateScheduler(2, 0.5, 0.1, 0.01);
        double rate = scheduler.nextRate(1.0, 1.0);
        rate = scheduler.nextRate(rate, 0.8);
        rate = scheduler.nextRate(rate, 0.8);
        near(1.0, rate, 0.0, "Geduld noch nicht erreicht");
        rate = scheduler.nextRate(rate, 0.8);
        near(0.5, rate, 0.0, "Plateau halbiert die Lernrate");
        rate = scheduler.nextRate(rate, 0.5);
        rate = scheduler.nextRate(rate, 0.5);
        near(0.5, rate, 0.0, "Verbesserung setzt Zähler zurück");
        for (int i = 0; i < 20; i++) {
            rate = scheduler.nextRate(rate, 0.5);
        }
        near(0.1, rate, 0.0, "Untere Grenze der Lernrate");
        pass("Lernratenanpassung: Plateau, Verbesserung und Untergrenze");
    }
    
    private static void testXor(double momentum, boolean adaptive) {
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 2, 4, 1 }, ActivationFunction.SIGMOID, 42L);
        double[][] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] targets = { { 0 }, { 1 }, { 1 }, { 0 } };
        LearningRateScheduler scheduler = new LearningRateScheduler(200, 0.5, 0.01, 0.00001);
        double rate = 1.0;
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.feedForward(inputs[i]);
                net.train(rate, targets[i], momentum);
            }
            if (adaptive) {
                double total = 0.0;
                for (int i = 0; i < inputs.length; i++) {
                    total += loss(net, inputs[i], targets[i]);
                }
                rate = scheduler.nextRate(rate, total / inputs.length);
            }
        }
        double total = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            double output = net.feedForward(inputs[i])[0];
            check((output >= 0.5 ? 1 : 0) == (int) targets[i][0], "XOR-Klasse für Muster " + i);
            total += loss(net, inputs[i], targets[i]);
        }
        check(total / inputs.length < 0.001, "Mittlerer XOR-Fehler E muss unter 0.001 liegen");
        pass("XOR lernt alle vier Muster: Momentum=" + momentum + ", adaptive=" + adaptive);
    }
    
    private static void testValidation() {
        rejects(IllegalArgumentException.class, () -> new FFNeuralNetwork(new int[]{ 2, 0, 1 }, ActivationFunction.SIGMOID));
        rejects(IllegalArgumentException.class, () -> new FFNeuralNetwork(new int[]{ 1, 1 }, null));
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 1, 1 }, ActivationFunction.LINEAR, 42L);
        rejects(IllegalStateException.class, () -> net.train(0.1, new double[]{ 0 }));
        rejects(IllegalArgumentException.class, () -> net.feedForward(null));
        rejects(IllegalArgumentException.class, () -> net.feedForward(new double[]{ Double.NaN }));
        rejects(IllegalArgumentException.class, () -> net.feedForward(new double[]{ 1, 2 }));
        net.feedForward(new double[]{ 1 });
        rejects(IllegalArgumentException.class, () -> net.train(0.1, new double[]{ 1, 2 }));
        rejects(IllegalArgumentException.class, () -> net.train(-1.0, new double[]{ 0 }));
        rejects(IllegalArgumentException.class, () -> net.train(0.1, new double[]{ 0 }, 1.0));
        rejects(IllegalArgumentException.class, () -> net.train(0.1, new double[]{ Double.NaN }));
        net.train(0.1, new double[]{ 0 });
        rejects(IllegalStateException.class, () -> net.train(0.1, new double[]{ 0 }));
        rejects(IllegalArgumentException.class, () -> CostFunction.MeanSquaredError.calculate(new double[0], new double[0]));
        rejects(IllegalArgumentException.class, () -> new LearningRateScheduler(0, 0.5, 0.01, 0.0));
        rejects(IllegalArgumentException.class, () -> new LearningRateScheduler(1, 1.0, 0.01, 0.0));
        LearningRateScheduler scheduler = new LearningRateScheduler(2, 0.5, 0.1, 0.0);
        rejects(IllegalArgumentException.class, () -> scheduler.nextRate(1.0, Double.NaN));
        pass("Ungültige Eingaben und Training ohne neuen Forward-Pass");
    }
    
    private static void testCsv() throws Exception {
        check(App.readData(Path.of("data/xor.csv")).length == 4, "Vier XOR-Zeilen laden");
        check(App.readData(Path.of("data/and.csv")).length == 4, "Vier AND-Zeilen laden");
        // System-Temp-Verzeichnis statt "bin/", damit der Test sowohl per
        // `java -cp bin ...` als auch in IntelliJ (target/classes) läuft.
        Path file = Files.createTempFile("invalid-data-", ".csv");
        try {
            for (String content : new String[]{ "a,b,c\n0,0,0", "x1,x2,target\n", "x1,x2,target\n0,1", "x1,x2,target\n0,1,2", "x1,x2,target\nNaN,1,0", "x1,x2,target\n0,?,0" }) {
                Files.writeString(file, content, StandardCharsets.UTF_8);
                rejects(IllegalArgumentException.class, () -> App.readData(file));
            }
        } finally {
            Files.deleteIfExists(file);
        }
        pass("CSV: Beispieldaten und fehlerhafte Zeilen");
    }
    
    // Nur im Test: Zugriff auf Parameter, ohne dafür die öffentliche Netz-API zu vergrößern.
    private static Layer[] layersOf(FFNeuralNetwork net) throws Exception {
        Field field = FFNeuralNetwork.class.getDeclaredField("layers");
        field.setAccessible(true);
        return (Layer[]) field.get(net);
    }
    
    private static double loss(FFNeuralNetwork net, double[] input, double[] target) {
        return CostFunction.HalfSquaredError.calculate(net.feedForward(input), target);
    }
    
    private static void near(double expected, double actual, double tolerance, String message) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tolerance, message + ": erwartet " + expected + ", erhalten " + actual);
    }
    
    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
    
    private static void rejects(Class<? extends Throwable> type, ThrowingAction action) {
        try {
            action.run();
        } catch (Throwable error) {
            if (type.isInstance(error)) {
                return;
            }
            throw new AssertionError("Erwartet: " + type.getSimpleName() + ", erhalten: " + error, error);
        }
        throw new AssertionError("Erwartete Ausnahme: " + type.getSimpleName());
    }
    
    private static void pass(String description) {
        passed++;
        System.out.println("OK: " + description);
    }
    
    @FunctionalInterface
    private interface ThrowingAction {
        void run() throws Exception;
    }
}
