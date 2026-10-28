package NeuralNetwork;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Kleine Konsolen-Demo: zwei Eingaben, vier Hidden-Neuronen, ein binärer Zielwert.
 * <p>
 * Aufruf: {@code java -cp bin NeuralNetwork.App [CSV-Datei] [sgd|momentum|adaptive] [Epochen]}.
 * Ohne Argumente werden die Standardwerte {@code data/xor.csv}, {@code sgd} und
 * {@code 10000} verwendet.
 */
public class App {
    
    /**
     * Liest eine CSV-Datei mit Trainingsdaten im Format {@code x1,x2,target}.
     * Die erste Zeile muss exakt der Kopfzeile entsprechen; leere Zeilen werden
     * ignoriert. Zielwerte müssen 0 oder 1 sein, alle Werte endlich.
     *
     * @param file Pfad zur CSV-Datei.
     * @return Array mit einer Zeile pro Trainingsmuster; Spalten 0 und 1 sind die
     * Eingaben, Spalte 2 ist der Zielwert.
     * @throws IOException              Wenn die Datei nicht gelesen werden kann.
     * @throws IllegalArgumentException Bei ungültigem Kopf, fehlerhaften Zeilen oder leerer Datei.
     */
    
    static double[][] readData(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty() || !lines.get(0).replace("\uFEFF", "").trim().equals("x1,x2,target")) {
            throw new IllegalArgumentException("CSV-Kopf muss x1,x2,target sein.");
        }
        List<double[]> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) {
                continue;
            }
            String[] cells = lines.get(i).split(",", -1);
            if (cells.length != 3) {
                throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": drei Zahlen erwartet.");
            }
            double[] row = new double[3];
            for (int j = 0; j < 3; j++) {
                try {
                    row[j] = Double.parseDouble(cells[j].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": ungültige Zahl.");
                }
                if (!Double.isFinite(row[j])) {
                    throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": Zahl muss endlich sein.");
                }
            }
            if (row[2] != 0.0 && row[2] != 1.0) {
                throw new IllegalArgumentException("CSV-Zeile " + (i + 1) + ": Zielwert muss 0 oder 1 sein.");
            }
            rows.add(row);
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("Die CSV-Datei enthält keine Trainingsmuster.");
        }
        return rows.toArray(new double[0][]);
    }
    
    /**
     * Einstiegspunkt. Delegiert an {@link #run(String[])} und fängt Fehler ab,
     * um eine verständliche Fehlermeldung samt Aufrufhilfe auszugeben.
     *
     * @param args Kommandozeilenargumente (siehe Klassen-Javadoc).
     */
    
    public static void main(String[] args) {
        try {
            run(args);
        } catch (IllegalArgumentException | IOException e) {
            System.err.println("Fehler: " + e.getMessage());
            System.err.println("Aufruf: java -cp bin NeuralNetwork.App [CSV-Datei] [sgd|momentum|adaptive] [Epochen]");
            System.exit(1);
        }
    }
    
    /**
     * Führt die komplette Demo aus: Daten laden, Netz aufbauen, trainieren
     * und Klassifikationsergebnisse ausgeben.
     *
     * @param args Bereits validierte Kommandozeilenargumente.
     * @throws IOException              Wenn die CSV-Datei nicht gelesen werden kann.
     * @throws IllegalArgumentException Bei ungültigen Argumenten, Daten oder instabilem Training.
     */
    
    private static void run(String[] args) throws IOException {
        if (args.length > 3) {
            throw new IllegalArgumentException("Zu viele Argumente.");
        }
        Path file = Path.of(args.length > 0 ? args[0] : "data/xor.csv");
        String mode = args.length > 1 ? args[1] : "sgd";
        int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 10000;
        if (!mode.equals("sgd") && !mode.equals("momentum") && !mode.equals("adaptive")) {
            throw new IllegalArgumentException("Unbekannter Modus: " + mode);
        }
        if (epochs < 1) {
            throw new IllegalArgumentException("Die Epochenzahl muss positiv sein.");
        }
        double[][] data = readData(file);
        FFNeuralNetwork net = new FFNeuralNetwork(new int[]{ 2, 4, 1 }, ActivationFunction.SIGMOID, 42L);
        double learningRate = 1.0;
        double momentum = mode.equals("sgd") ? 0.0 : 0.9;
        LearningRateScheduler scheduler = new LearningRateScheduler(200, 0.5, 0.01, 0.00001);
        
        System.out.printf("Datei: %s | Muster: %d | Netz: 2-4-1 | Seed: 42%n", file, data.length);
        System.out.printf("Modus: %s | Momentum: %.1f | Epochen: %d%n", mode, momentum, epochs);
        System.out.printf(Locale.ROOT, "Mittlerer Fehler E vor Training: %.6f%n", meanLoss(net, data));
        
        long startTime = System.nanoTime();
        boolean reachedThreshold = false;
        long epochAtThreshold = -1;
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Online-Lernen: ein Update pro Datenzeile, feste Reihenfolge für die Demo.
            for (double[] row : data) {
                net.feedForward(new double[]{ row[0], row[1] });
                net.train(learningRate, new double[]{ row[2] }, momentum);
            }
            // Alle Muster erneut auswerten: einheitlicher Gewichtsstand für den Epochenfehler.
            double loss = meanLoss(net, data);
            if (!Double.isFinite(loss)) {
                throw new IllegalArgumentException("Training instabil. Eingaben auf einen kleinen Wertebereich skalieren.");
            }
            if (!reachedThreshold && loss < 0.01) {
                reachedThreshold = true;
                epochAtThreshold = epoch;
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
        long endTime = System.nanoTime();
        System.out.printf(Locale.ROOT, "\nLaufzeit: %.2f ms%n", (endTime - startTime) / 1_000_000.0);
        System.out.printf("Schwelle E<0,01 erreicht bei Epoche: %d%n", epochAtThreshold);
        System.out.println("\nx1, x2 -> Ausgabe | Klasse | Ziel");
        int correct = 0;
        for (double[] row : data) {
            double output = net.feedForward(new double[]{ row[0], row[1] })[0];
            int predicted = output >= 0.5 ? 1 : 0;
            if (predicted == (int) row[2]) {
                correct++;
            }
            System.out.printf(Locale.ROOT, "%.2f, %.2f -> %.4f | %d | %.0f%n", row[0], row[1], output, predicted, row[2]);
        }
        System.out.printf("Korrekt auf den Trainingsmustern: %d/%d%n", correct, data.length);
    }
    
    /**
     * Berechnet den mittleren Fehler (halbe Fehlerquadratsumme) über alle Muster
     * bei unveränderten Gewichten. Wird zur Fortschrittsanzeige genutzt; das
     * Training selbst verwendet den Fehler nicht direkt.
     *
     * @param net  Das trainierte Netz.
     * @param data Die Trainingsmuster.
     * @return Der mittlere Fehler pro Muster.
     */
    
    private static double meanLoss(FFNeuralNetwork net, double[][] data) {
        double total = 0.0;
        for (double[] row : data) {
            total += CostFunction.HalfSquaredError.calculate(net.feedForward(new double[]{ row[0], row[1] }), new double[]{ row[2] });
        }
        return total / data.length;
    }
}
