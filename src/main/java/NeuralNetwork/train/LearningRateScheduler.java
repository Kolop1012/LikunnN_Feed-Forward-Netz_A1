package NeuralNetwork.train;

/**
 * Reduziert bei einem Fehlerplateau die Lernrate bis zur angegebenen Untergrenze.
 */
public class LearningRateScheduler {
    private final int patience;
    private final double factor;
    private final double minLearningRate;
    private final double minImprovement;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private int epochsWithoutImprovement;
    
    /**
     * Erstellt einen neuen Scheduler.
     *
     * @param patience        Anzahl der Epochen ohne ausreichende Verbesserung, nach
     *                        denen die Lernrate reduziert wird.
     * @param factor          Multiplikationsfaktor für die Reduktion (z. B. {@code 0.5}
     *                        für Halbierung). Muss in {@code (0, 1)} liegen.
     * @param minLearningRate Untergrenze, unter die die Lernrate nicht sinkt.
     * @param minImprovement  Mindestverbesserung des Fehlers, damit eine Epoche
     *                        als Fortschritt zählt.
     * @throws IllegalArgumentException Bei ungültigen Parametern (z. B. {@code patience < 1},
     *                                  {@code factor} außerhalb {@code (0, 1)}).
     */
    public LearningRateScheduler(int patience, double factor, double minLearningRate, double minImprovement) {
        if (patience < 1 || !Double.isFinite(factor) || factor <= 0.0 || factor >= 1.0 || !Double.isFinite(minLearningRate) || minLearningRate <= 0.0 || !Double.isFinite(minImprovement) || minImprovement < 0.0) {
            throw new IllegalArgumentException("Ungültige Parameter für die Lernratenanpassung.");
        }
        this.patience = patience;
        this.factor = factor;
        this.minLearningRate = minLearningRate;
        this.minImprovement = minImprovement;
    }
    
    /**
     * Berechnet die Lernrate für die nächste Epoche. Wird einmal pro Epoche mit
     * dem aktuellen Fehler bei unveränderten Gewichten aufgerufen. Verbessert
     * sich der Fehler nicht um mindestens {@code minImprovement}, wird nach
     * {@code patience} Epochen die Lernrate mit {@code factor} multipliziert
     * (jedoch nicht unter {@code minLearningRate}).
     *
     * @param currentRate Aktuelle Lernrate ({@code > 0}).
     * @param loss        Aktueller Fehler ({@code >= 0}).
     * @return Die Lernrate für die nächste Epoche.
     * @throws IllegalArgumentException Wenn {@code currentRate} oder {@code loss}
     *                                  ungültig oder nicht endlich sind.
     */
    public double nextRate(double currentRate, double loss) {
        if (!Double.isFinite(currentRate) || currentRate < minLearningRate || !Double.isFinite(loss) || loss < 0.0) {
            throw new IllegalArgumentException("Lernrate und Fehler müssen gültige endliche Werte sein.");
        }
        if (loss < bestLoss - minImprovement) {
            bestLoss = loss;
            epochsWithoutImprovement = 0;
        } else if (++epochsWithoutImprovement >= patience) {
            epochsWithoutImprovement = 0;
            return Math.max(minLearningRate, currentRate * factor);
        }
        return currentRate;
    }
}
