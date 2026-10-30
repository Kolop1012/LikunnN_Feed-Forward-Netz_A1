package NeuralNetwork;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.nio.file.InvalidPathException;
import java.util.Arrays;

/** Kleine Konsolen-Demo mit dynamischer Input- und Output-Skalierung. */
public class App {

    public record AppConfig(double[][] data, String mode, int epochs, int[] indexausgabewerte, int inputneuronen, int neuronenhiddenlayer, int outputneuronen){}
    public static AppConfig validateArgs(String[] args) throws IOException {
        if (args.length > 5) {
            throw new IllegalArgumentException("Zu viele Argumente.");
        }

        Path pathdata = validateAndGetPath(args.length > 0 ? args[0] : "data/data_cats_modified.csv");
        double[][] data = readData(pathdata);
        String mode = args.length > 1 ? args[1] : "sgd";
        if (!mode.equals("sgd") && !mode.equals("momentum") && !mode.equals("adaptive")) {
            throw new IllegalArgumentException("Unbekannter Modus: " + mode);
        }
        int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 10000;
        if (epochs < 1) {
            throw new IllegalArgumentException("Die Epochenzahl muss positiv sein.");
        }

        int inputneuronen;
        int outputneuronen;
        int[] indexausgabewerte;

        if (args.length > 3) {
            if (!args[3].matches("^\\d+-\\d+$")) {
                throw new IllegalArgumentException("Output-Spalten ungültig angegeben: '" + args[3] + "' (Erwartetes Format: 'min-max', z. B. '3-5')");
            }
            String[] parts = args[3].split("-");
            indexausgabewerte = new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
            outputneuronen = indexausgabewerte[1] - indexausgabewerte[0] + 1;
            inputneuronen = data[0].length - outputneuronen;
        } else {
            indexausgabewerte = new int[]{2, 2};
            outputneuronen = 1;
            inputneuronen = 2;
        }

        int neuronenhiddenlayer = 4;
        if (args.length > 4) {
            neuronenhiddenlayer = Integer.parseInt(args[4]);
            if (neuronenhiddenlayer < 1) {
                throw new IllegalArgumentException("Mindestens ein Neuron pro Hiddenlayer benötigt");
            }
        }
        return new AppConfig(data, mode, epochs, indexausgabewerte, inputneuronen, neuronenhiddenlayer, outputneuronen);
    }

    public static void main(String[] args) {
        // Nur debugging
        args = new String[]{"data/data_cats_modified.csv", "sgd", "1000", "1-1", "2"};

        try {
            AppConfig argumente = validateArgs(args);
            run(argumente);
        } catch (IllegalArgumentException | IOException e) {
            System.err.println("Fehler: " + e.getMessage());
            System.err.println("Aufruf: java -cp bin NeuralNetwork.App [CSV-Datei] [sgd|momentum|adaptive] [Epochen]");
            System.exit(1);
        }
    }

    private static void run(AppConfig argumente) throws IOException {
        double[][] data = argumente.data;
        String mode = argumente.mode;
        int epochs = argumente.epochs;
        int[] indexausgabewerte = argumente.indexausgabewerte;
        int inputneuronen = argumente.inputneuronen;
        int neuronenhiddenlayer = argumente.neuronenhiddenlayer;
        int outputneuronen = argumente.outputneuronen;

        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{inputneuronen, neuronenhiddenlayer, outputneuronen}, ActivationFunction.SIGMOID, 42L);
        double learningRate = 1.0;
        double momentum = mode.equals("sgd") ? 0.0 : 0.9;
        LearningRateScheduler scheduler = new LearningRateScheduler(200, 0.5, 0.01, 0.00001);

        // --- 1. Min/Max für Inputs ermitteln ---
        double[] inputMins = new double[inputneuronen];
        double[] inputMaxes = new double[inputneuronen];
        Arrays.fill(inputMins, Double.POSITIVE_INFINITY);
        Arrays.fill(inputMaxes, Double.NEGATIVE_INFINITY);

        for (double[] row : data) {
            double[] in = extractInputs(row, indexausgabewerte, inputneuronen);
            for (int k = 0; k < inputneuronen; k++) {
                if (in[k] < inputMins[k]) inputMins[k] = in[k];
                if (in[k] > inputMaxes[k]) inputMaxes[k] = in[k];
            }
        }

        // --- 2. Min/Max für Targets ermitteln ---
        double[] targetMins = new double[outputneuronen];
        double[] targetMaxes = new double[outputneuronen];
        Arrays.fill(targetMins, Double.POSITIVE_INFINITY);
        Arrays.fill(targetMaxes, Double.NEGATIVE_INFINITY);

        for (double[] row : data) {
            double[] t = extractTargets(row, indexausgabewerte, outputneuronen);
            for (int k = 0; k < outputneuronen; k++) {
                if (t[k] < targetMins[k]) targetMins[k] = t[k];
                if (t[k] > targetMaxes[k]) targetMaxes[k] = t[k];
            }
        }

        System.out.printf("Muster: %d | Netz: %d-%d-%d | Seed: 42%n", data.length, inputneuronen, neuronenhiddenlayer, outputneuronen);
        System.out.printf("Modus: %s | Momentum: %.1f | Epochen: %d%n", mode, momentum, epochs);
        System.out.printf(Locale.ROOT, "Mittlerer Fehler E vor Training: %.6f%n",
                meanLoss(net, data, indexausgabewerte, inputneuronen, outputneuronen, inputMins, inputMaxes, targetMins, targetMaxes));

        List<Double> lossHistory = new ArrayList<>();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Online-Lernen
            for (double[] row : data) {
                double[] inputs = extractInputs(row, indexausgabewerte, inputneuronen);
                double[] targets = extractTargets(row, indexausgabewerte, outputneuronen);

                // Eingaben und Zielwerte auf [0, 1] skalieren
                for (int k = 0; k < inputneuronen; k++) {
                    inputs[k] = scale(inputs[k], inputMins[k], inputMaxes[k]);
                }
                for (int k = 0; k < outputneuronen; k++) {
                    targets[k] = scale(targets[k], targetMins[k], targetMaxes[k]);
                }

                net.feedForward(inputs);
                net.train(learningRate, targets, momentum);
            }

            double loss = meanLoss(net, data, indexausgabewerte, inputneuronen, outputneuronen, inputMins, inputMaxes, targetMins, targetMaxes);
            lossHistory.add(loss);

            if (!Double.isFinite(loss)) {
                throw new IllegalArgumentException("Training instabil. Eingaben auf einen kleinen Wertebereich skalieren.");
            }
            if (epoch == 1 || epoch % 1000 == 0 || epoch == epochs) {
                System.out.printf(Locale.ROOT, "Epoche %5d | Fehler E %.6f | Lernrate %.5f%n", epoch, loss, learningRate);
            }
            if (mode.equals("adaptive")) {
                double nextRate = scheduler.nextRate(learningRate, loss);
                if (nextRate < learningRate) {
                    System.out.printf(Locale.ROOT, "  Nach Epoche %d: Lernrate %.5f -> %.5f%n", epoch, learningRate, nextRate);
                }
                learningRate = nextRate;
            }
        }

        System.out.println("\nEingaben -> Ausgaben | Vorhergesagt | Ziel");
        int correct = 0;
        for (double[] row : data) {
            double[] rawInputs = extractInputs(row, indexausgabewerte, inputneuronen);
            double[] scaledInputs = new double[inputneuronen];
            for (int k = 0; k < inputneuronen; k++) {
                scaledInputs[k] = scale(rawInputs[k], inputMins[k], inputMaxes[k]);
            }

            double[] originalTargets = extractTargets(row, indexausgabewerte, outputneuronen);

            // Vorwärtsdurchlauf mit skalierten Inputs
            double[] rawOutputs = net.feedForward(scaledInputs);
            double[] finalOutputs = new double[outputneuronen];

            boolean isCorrect = true;
            for (int k = 0; k < outputneuronen; k++) {
                // Ausgaben auf ursprünglichen Wertebereich hochskalieren
                finalOutputs[k] = unscale(rawOutputs[k], targetMins[k], targetMaxes[k]);

                long predictedClass = Math.round(finalOutputs[k]);
                long actualClass = Math.round(originalTargets[k]);

                if (predictedClass != actualClass) {
                    isCorrect = false;
                }
            }
            if (isCorrect) {
                correct++;
            }

            System.out.printf(Locale.ROOT, "%s -> %s | Ziel: %s%n",
                    Arrays.toString(rawInputs), Arrays.toString(finalOutputs), Arrays.toString(originalTargets));
        }
        System.out.printf("Korrekt auf den Trainingsmustern: %d/%d%n", correct, data.length);

        double[][][] finalWeights = net.getWeights();
        double[][] finalBiases = net.getBiases();
        NetworkVisualizer.saveVisualization(
                "grafik_auswertung.png", lossHistory,
                new int[]{inputneuronen, neuronenhiddenlayer, outputneuronen},
                finalWeights, finalBiases, mode, learningRate, epochs, correct, data.length
        );
    }

    private static double[] extractInputs(double[] row, int[] outputIndices, int expectedInputs) {
        double[] inputs = new double[expectedInputs];
        int inputIndex = 0;
        for (int col = 0; col < row.length; col++) {
            if (!isOutputColumn(col, outputIndices)) {
                inputs[inputIndex++] = row[col];
            }
        }
        return inputs;
    }

    private static double[] extractTargets(double[] row, int[] outputIndices, int expectedOutputs) {
        double[] targets = new double[expectedOutputs];
        int startCol = outputIndices[0];
        for (int i = 0; i < expectedOutputs; i++) {
            targets[i] = row[startCol + i];
        }
        return targets;
    }

    private static boolean isOutputColumn(int col, int[] outputIndices) {
        return col >= outputIndices[0] && col <= outputIndices[1];
    }

    static double[][] readData(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);

        if (lines.isEmpty()) {
            throw new IllegalArgumentException("Datei ist leer.");
        }
        if (lines.size() < 2) {
            throw new IllegalArgumentException("Die CSV-Datei enthält keine Trainingsmuster.");
        }

        List<double[]> rows = new ArrayList<>();
        int spaltenanzahl = lines.get(1).split("[,;]", -1).length;

        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) {
                continue;
            }

            String[] cells = lines.get(i).split("[,;]", -1);
            if (cells.length != spaltenanzahl) {
                throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": " + spaltenanzahl + " Zahlen erwartet.");
            }

            double[] row = new double[spaltenanzahl];
            for (int j = 0; j < spaltenanzahl; j++) {
                try {
                    row[j] = Double.parseDouble(cells[j].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": ungültige Zahl.");
                }
                if (!Double.isFinite(row[j])) {
                    throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": Zahl muss endlich sein.");
                }
            }

            rows.add(row);
        }

        if (rows.isEmpty()) {
            throw new IllegalArgumentException("Die CSV-Datei enthält keine gültigen Trainingsmuster.");
        }

        return rows.toArray(new double[0][]);
    }

    private static double meanLoss(FFNeuralNetwork net, double[][] data, int[] indexausgabewerte, int inputneuronen, int outputneuronen, double[] inputMins, double[] inputMaxes, double[] targetMins, double[] targetMaxes) {
        double total = 0.0;
        for (double[] row : data) {
            double[] inputs = extractInputs(row, indexausgabewerte, inputneuronen);
            double[] targets = extractTargets(row, indexausgabewerte, outputneuronen);

            for (int k = 0; k < inputneuronen; k++) {
                inputs[k] = scale(inputs[k], inputMins[k], inputMaxes[k]);
            }
            for (int k = 0; k < outputneuronen; k++) {
                targets[k] = scale(targets[k], targetMins[k], targetMaxes[k]);
            }

            total += CostFunction.HalfSquaredError.calculate(
                    net.feedForward(inputs), targets);
        }
        return total / data.length;
    }

    private static double scale(double val, double min, double max) {
        if (min == max) return 0.5;
        return (val - min) / (max - min);
    }

    private static double unscale(double val, double min, double max) {
        if (min == max) return min;
        return val * (max - min) + min;
    }

    private static Path validateAndGetPath(String rawPath) {
        Path path;

        try {
            path = Path.of(rawPath);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("Der eingegebene Pfad enthält ungültige Zeichen: " + rawPath);
        }

        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Datei nicht gefunden: " + path.toAbsolutePath());
        }

        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Der Pfad verweist auf einen Ordner, keine Datei: " + path);
        }

        if (!Files.isReadable(path)) {
            throw new IllegalArgumentException("Keine Leserechte für die Datei: " + path);
        }

        return path;
    }
}