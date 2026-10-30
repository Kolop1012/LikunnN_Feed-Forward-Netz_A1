package NeuralNetwork.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class CsvParser {
    
    public static double[][] parse(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8)
                                  .stream()
                                  .map(line -> line.replace("\uFEFF", "")
                                                   .trim())
                                  .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                                  .toList();
        
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("Die CSV-Datei enthält keine Daten.");
        }
        
        String separator = lines.getFirst()
                                .contains(";") ? ";" : ",";
        String[] firstCells = lines.getFirst()
                                   .split(separator, -1);
        boolean hasHeader = !isNumericRow(firstCells);
        
        if (hasHeader && lines.size() < 2) {
            throw new IllegalArgumentException("Die CSV-Datei braucht Header und mindestens ein Trainingsmuster.");
        }
        
        int expectedCols = hasHeader ? firstCells.length : -1;
        
        double[][] data = lines.stream()
                               .skip(hasHeader ? 1 : 0)
                               .map(line -> line.split(separator, -1))
                               .map(cells -> Arrays.stream(cells)
                                                   .mapToDouble(s -> parseFinite(s.trim()))
                                                   .toArray())
                               .toArray(double[][]::new);
        
        if (data.length == 0) {
            throw new IllegalArgumentException("Die CSV-Datei enthält keine Trainingsmuster.");
        }
        
        int nCols = data[0].length;
        for (int i = 0; i < data.length; i++) {
            if (data[i].length != nCols) {
                throw new IllegalArgumentException(
                        "Zeile " + (i + 2 + (hasHeader ? 1 : 0)) + ": " + nCols + " Spalten erwartet, " + data[i].length + " gefunden.");
            }
            if (expectedCols > 0 && data[i].length != expectedCols) {
                throw new IllegalArgumentException(
                        "Zeile " + (i + 2 + 1) + ": Header hat " + expectedCols + " Spalten, Datenzeile " + data[i].length + ".");
            }
        }
        return data;
    }
    
    /**
     * Transformiert die Spalte {@code classColumn} in One-Hot-Kodierung.
     * Die Klassen werden sortiert, die ursprünglichen Labels werden zurückgegeben.
     *
     * @return Array mit [transformierten Daten, sortierten Klassenlabels]
     */
    public static OneHotResult toOneHot(double[][] data, int classColumn) {
        if (classColumn < 0 || classColumn >= data[0].length) {
            throw new IllegalArgumentException("Klassenspalte " + classColumn + " außerhalb 0-" + (data[0].length - 1));
        }
        
        // Eindeutige Klassen sortiert sammeln
        TreeSet<Double> classes = Arrays.stream(data)
                                        .map(row -> row[classColumn])
                                        .collect(Collectors.toCollection(TreeSet::new));
        int nClasses = classes.size();
        if (nClasses < 2) {
            throw new IllegalArgumentException("One-Hot benötigt mindestens 2 verschiedene Klassen.");
        }
        Map<Double, Integer> classToIdx = new HashMap<>();
        int idx = 0;
        for (double c : classes)
            classToIdx.put(c, idx++);
        
        int newCols = data[0].length - 1 + nClasses;
        double[][] result = new double[data.length][newCols];
        for (int i = 0; i < data.length; i++) {
            int out = 0;
            for (int j = 0; j < data[0].length; j++) {
                if (j == classColumn) {
                    int classIdx = classToIdx.get(data[i][j]);
                    for (int k = 0; k < nClasses; k++) {
                        result[i][out++] = (k == classIdx) ? 1.0 : 0.0;
                    }
                } else {
                    result[i][out++] = data[i][j];
                }
            }
        }
        
        double[] labels = classes.stream()
                                 .mapToDouble(Double::doubleValue)
                                 .toArray();
        return new OneHotResult(result, labels);
    }
    
    public static Path validateAndGetPath(String rawPath) {
        Path path;
        try {
            path = Path.of(rawPath);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException("Ungültiger Pfad: " + rawPath);
        }
        if (!Files.exists(path))
            throw new IllegalArgumentException("Datei nicht gefunden: " + path.toAbsolutePath());
        if (!Files.isRegularFile(path))
            throw new IllegalArgumentException("Kein reguläres File: " + path);
        if (!Files.isReadable(path))
            throw new IllegalArgumentException("Keine Leserechte: " + path);
        return path;
    }
    
    private static boolean isNumericRow(String[] cells) {
        for (String cell : cells) {
            try {
                if (!Double.isFinite(Double.parseDouble(cell.trim())))
                    return false;
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }
    
    private static double parseFinite(String s) {
        double v;
        try {
            v = Double.parseDouble(s);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Ungültige Zahl: '" + s + "'");
        }
        if (!Double.isFinite(v))
            throw new IllegalArgumentException("Nicht-endliche Zahl: " + s);
        return v;
    }
    
    public record OneHotResult(double[][] data, double[] classLabels) {
    }
}
