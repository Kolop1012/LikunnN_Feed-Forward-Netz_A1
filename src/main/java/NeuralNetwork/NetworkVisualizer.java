package NeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

public class NetworkVisualizer {

    public static void saveVisualization(
            String filename,
            List<Double> lossHistory,
            int[] networkLayers,
            double[][][] weights,
            double[][] biases,
            String mode,
            double learningRate,
            int totalEpochs,
            int correctSamples,
            int totalSamples
    ) throws IOException {
        int width = 850;
        int height = 1100;

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();

        // Hochwertiges Rendering
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        // Weißer Hintergrund
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);

        // 1. Plot zeichnen
        drawLossPlot(g, lossHistory, 120, 80, 600, 300, mode, learningRate, totalEpochs);

        // 2. Netzwerk zeichnen (Mit dynamischer Skalierung)
        drawNetworkGraph(g, networkLayers, weights, biases, 120, 540, 600, 450, correctSamples, totalSamples);

        g.dispose();
        ImageIO.write(image, "png", new File(filename));
        System.out.println("Visualisierung erfolgreich gespeichert unter: " + filename);
    }

    private static void drawLossPlot(Graphics2D g, List<Double> lossHistory, int x, int y, int w, int h, String mode, double lr, int totalEpochs) {
        g.setFont(new Font("Serif", Font.PLAIN, 14));

        double minLog = -4.5;
        double maxLog = -0.5;

        // --- Y-Achse (Logarithmisch) ---
        for (int exp = -5; exp <= 0; exp++) {
            for (int m = 1; m <= 9; m++) {
                double val = m * Math.pow(10, exp);
                double logVal = Math.log10(val);

                if (logVal >= minLog && logVal <= maxLog) {
                    int yPos = y + h - (int) ((logVal - minLog) / (maxLog - minLog) * h);

                    if (m == 1) { // Major Tick
                        g.setColor(new Color(180, 180, 180));
                        g.drawLine(x, yPos, x + w, yPos);
                        g.setColor(Color.BLACK);
                        g.drawString("10^" + exp, x - 40, yPos + 5);
                    } else { // Minor Tick
                        g.setColor(new Color(235, 235, 235));
                        g.drawLine(x, yPos, x + w, yPos);
                    }
                }
            }
        }

        // --- X-Achse (Dynamisch skaliert) ---
        int maxEpoch = Math.max(1, lossHistory.size() - 1);
        int exponent = (int) Math.log10(maxEpoch);
        double divisor = Math.pow(10, exponent);

        for (int i = 0; i <= 10; i++) {
            int xPos = x + (int) ((i / 10.0) * w);
            g.setColor(new Color(180, 180, 180));
            g.drawLine(xPos, y, xPos, y + h);
            g.setColor(Color.BLACK);

            double tickVal = (maxEpoch * (i / 10.0)) / divisor;
            String tickLabel = (exponent > 0) ? String.format(Locale.US, "%.1f", tickVal) : String.format(Locale.US, "%.0f", tickVal);
            g.drawString(tickLabel, xPos - 10, y + h + 20);
        }

        g.setColor(Color.BLACK);
        g.setStroke(new BasicStroke(1.0f));
        g.drawRect(x, y, w, h);

        g.drawString("Epoche", x + w / 2 - 20, y + h + 45);
        if (exponent > 0) {
            g.setFont(new Font("Serif", Font.BOLD, 14));
            g.drawString("·10^" + exponent, x + w - 10, y + h + 45);
        }

        g.setFont(new Font("Serif", Font.PLAIN, 14));
        Graphics2D g2 = (Graphics2D) g.create();
        g2.rotate(-Math.PI / 2, x - 55, y + h / 2 + 50);
        g2.drawString("Mittlerer Fehler E", x - 55, y + h / 2 + 50);
        g2.dispose();

        // --- Loss-Kurve ---
        if (lossHistory != null && !lossHistory.isEmpty()) {
            g.setColor(Color.BLUE);
            g.setStroke(new BasicStroke(1.5f));

            int points = lossHistory.size();
            for (int i = 0; i < points - 1; i++) {
                double v1 = Math.log10(Math.max(lossHistory.get(i), 1e-10));
                double v2 = Math.log10(Math.max(lossHistory.get(i + 1), 1e-10));

                v1 = Math.max(minLog, Math.min(maxLog, v1));
                v2 = Math.max(minLog, Math.min(maxLog, v2));

                int x1 = x + (int) ((double) i / (points - 1) * w);
                int y1 = y + h - (int) ((v1 - minLog) / (maxLog - minLog) * h);
                int x2 = x + (int) ((double) (i + 1) / (points - 1) * w);
                int y2 = y + h - (int) ((v2 - minLog) / (maxLog - minLog) * h);

                g.drawLine(x1, y1, x2, y2);
            }
        }

        g.setStroke(new BasicStroke(1.0f));
        g.setColor(Color.WHITE);
        g.fillRect(x + w - 160, y + 10, 150, 25);
        g.setColor(Color.BLACK);
        g.drawRect(x + w - 160, y + 10, 150, 25);
        g.setColor(Color.BLUE);
        g.drawLine(x + w - 150, y + 22, x + w - 120, y + 22);
        g.setColor(Color.BLACK);
        g.drawString(mode.toUpperCase() + " (Lernrate " + lr + ")", x + w - 110, y + 27);

        String caption1 = String.format(Locale.GERMAN, "Abbildung 1: Lernverlauf (Loss History) für das Katzen-Problem über %,d Epochen im %s-Modus. Dargestellt", totalEpochs, mode.toUpperCase());
        String caption2 = "ist die halbe Fehlerquadratsumme.";
        g.drawString(caption1, x, y + h + 85);
        g.drawString(caption2, x, y + h + 105);
    }

    private static void drawNetworkGraph(Graphics2D g, int[] layers, double[][][] weights, double[][] biases, int x, int y, int w, int h, int correctSamples, int totalSamples) {
        int numLayers = layers.length;
        int layerSpacing = w / (numLayers - 1);

        // --- DYNAMISCHE SKALIERUNG ---
        // 1. Größte Schicht ermitteln
        int maxNodes = 0;
        for (int count : layers) {
            maxNodes = Math.max(maxNodes, count);
        }

        // 2. Maximaler vertikaler Abstand (capped bei 80px)
        int vGap = Math.min(80, (h - 40) / maxNodes);

        // 3. Radius der Neuronen abhängig vom Platz berechnen (min 4px, max 25px)
        int radius = Math.max(4, Math.min(25, (int)(vGap * 0.35)));

        Point[][] nodePositions = new Point[numLayers][];
        int centerY = y + h / 2;

        for (int l = 0; l < numLayers; l++) {
            int numNodes = layers[l];
            nodePositions[l] = new Point[numNodes];
            int startY = centerY - (numNodes - 1) * vGap / 2;
            for (int n = 0; n < numNodes; n++) {
                nodePositions[l][n] = new Point(x + l * layerSpacing, startY + n * vGap);
            }
        }

        // --- VERBINDUNGEN (Pfeile) ---
        g.setColor(new Color(60, 60, 60));
        for (int l = 0; l < numLayers - 1; l++) {
            for (int i = 0; i < layers[l]; i++) {
                for (int j = 0; j < layers[l + 1]; j++) {
                    Point p1 = nodePositions[l][i];
                    Point p2 = nodePositions[l + 1][j];

                    float thickness = 1.0f;
                    if (weights != null && weights.length > l && weights[l].length > i && weights[l][i].length > j) {
                        thickness = 0.5f + (float) Math.abs(weights[l][i][j]) * 0.35f;
                    }
                    // Deckelung der Liniendicke: Darf nicht dicker als die halbe Neuronengröße werden
                    thickness = Math.max(0.5f, Math.min(radius * 0.5f, Math.min(6.0f, thickness)));
                    drawArrow(g, p1.x, p1.y, p2.x, p2.y, radius, thickness);
                }
            }
        }

        // --- NEURONEN (Kreise) ---
        // Die Schriftgröße schrumpft dynamisch mit dem Radius mit
        int fontSize = Math.max(9, (int)(radius * 0.65));
        g.setFont(new Font("Serif", Font.ITALIC, fontSize));

        for (int l = 0; l < numLayers; l++) {
            for (int n = 0; n < layers[l]; n++) {
                Point p = nodePositions[l][n];

                float percentBlack = 0;
                if (l > 0 && biases != null && biases.length > l - 1 && biases[l - 1].length > n) {
                    double b = biases[l - 1][n];
                    percentBlack = (float) Math.max(0, Math.min(100, 40 - b * 10));
                } else if (l > 0) {
                    percentBlack = (l == numLayers - 1) ? 80 : 20;
                }

                int gray = (int) (255 * (100 - percentBlack) / 100);
                Color fillColor = new Color(gray, gray, gray);

                g.setColor(fillColor);
                g.fillOval(p.x - radius, p.y - radius, 2 * radius, 2 * radius);

                g.setColor(Color.BLACK);
                // Rahmendicke skaliert minimal mit dem Radius
                g.setStroke(new BasicStroke(Math.max(0.5f, radius * 0.05f)));
                g.drawOval(p.x - radius, p.y - radius, 2 * radius, 2 * radius);

                // Beschriftung nur einfügen, wenn die Kreise groß genug sind (Radius >= 10)
                if (radius >= 10) {
                    g.setColor(percentBlack > 50 ? Color.WHITE : Color.BLACK);

                    FontMetrics fm = g.getFontMetrics();
                    String label = "";
                    if (l == 0) label = "x" + (n + 1);
                    else if (l == numLayers - 1) label = "y" + (layers[l] > 1 ? String.valueOf(n + 1) : "");

                    int textX = p.x - fm.stringWidth(label) / 2;
                    int textY = p.y + fm.getAscent() / 2 - 2;
                    g.drawString(label, textX, textY);
                }
            }
        }

        // --- ERGEBNIS-TEXT ---
        g.setFont(new Font("Serif", Font.BOLD, 15));
        g.setColor(Color.BLACK);
        double rate = ((double) correctSamples / totalSamples) * 100.0;
        String accString = String.format(Locale.GERMAN, "Ergebnis: Das Netz erkannte %d aus %d Proben richtig, also mit einer Rate von %.1f Prozent.", correctSamples, totalSamples, rate);
        g.drawString(accString, x, y + h + 40);
    }

    private static void drawArrow(Graphics2D g, int x1, int y1, int x2, int y2, int radius, float thickness) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double dist = Math.sqrt(dx * dx + dy * dy);

        int sx = (int) (x1 + radius * dx / dist);
        int sy = (int) (y1 + radius * dy / dist);
        int ex = (int) (x2 - radius * dx / dist);
        int ey = (int) (y2 - radius * dy / dist);

        g.setStroke(new BasicStroke(thickness, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.drawLine(sx, sy, ex, ey);

        double angle = Math.atan2(dy, dx);
        // Die Pfeilspitze skaliert jetzt auch dynamisch, um bei Mini-Kreisen nicht absurd groß zu wirken
        int arrowSize = Math.max(4, Math.min(radius, (int) (thickness * 2.5)));

        Polygon head = new Polygon();
        head.addPoint(ex, ey);
        head.addPoint((int) (ex - arrowSize * Math.cos(angle - Math.PI / 6)), (int) (ey - arrowSize * Math.sin(angle - Math.PI / 6)));
        head.addPoint((int) (ex - arrowSize * Math.cos(angle + Math.PI / 6)), (int) (ey - arrowSize * Math.sin(angle + Math.PI / 6)));

        g.fill(head);
    }
}