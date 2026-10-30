package NeuralNetwork.config;

import NeuralNetwork.data.CsvParser;
import NeuralNetwork.data.Dataset;

import java.io.IOException;
import java.util.Set;

/**
 * Konfiguration aus CLI-Argumenten, inkl. One-Hot-Transformation.
 */
public record AppConfig(Dataset dataset, String mode, int epochs, int hiddenNeurons, boolean oneHot,
                        double[] classLabels) {
    
    public static AppConfig parse(String[] args) throws IOException {
        if (args.length > 6)
            throw new IllegalArgumentException("Zu viele Argumente.");
        
        double[][] data = CsvParser.parse(CsvParser.validateAndGetPath(args.length > 0 ? args[0] : "data/xor.csv"));
        
        String mode = args.length > 1 ? args[1] : "sgd";
        if (!Set.of("sgd", "momentum", "adaptive")
                .contains(mode)) {
            throw new IllegalArgumentException("Unbekannter Modus: " + mode);
        }
        
        int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 10000;
        if (epochs < 1)
            throw new IllegalArgumentException("Epochenzahl muss positiv sein.");
        
        int[] outRange = parseOutputRange(args.length > 3 ? args[3] : null, data[0].length);
        
        boolean oneHot = args.length > 5 && "one-hot".equalsIgnoreCase(args[5]);
        double[] labels = null;
        if (oneHot) {
            CsvParser.OneHotResult r = CsvParser.toOneHot(data, outRange[0]);
            data = r.data();
            labels = r.classLabels();
            outRange = new int[]{ data[0].length - labels.length, data[0].length - 1 };
        }
        
        int hidden = args.length > 4 ? Integer.parseInt(args[4]) : 4;
        if (hidden < 1)
            throw new IllegalArgumentException("Mindestens ein Hidden-Neuron.");
        
        return new AppConfig(new Dataset(data, outRange), mode, epochs, hidden, oneHot, labels);
    }
    
    private static int[] parseOutputRange(String spec, int cols) {
        if (spec == null)
            return new int[]{ cols - 1, cols - 1 };
        if (!spec.matches("\\d+-\\d+")) {
            throw new IllegalArgumentException("Output-Spalten ungültig: '" + spec + "'");
        }
        String[] p = spec.split("-");
        int min = Integer.parseInt(p[0]), max = Integer.parseInt(p[1]);
        if (min < 0 || max >= cols || min > max) {
            throw new IllegalArgumentException("Output-Spalten außerhalb 0-" + (cols - 1));
        }
        if (max - min + 1 >= cols) {
            throw new IllegalArgumentException("Mindestens eine Input-Spalte benötigt.");
        }
        return new int[]{ min, max };
    }
}
