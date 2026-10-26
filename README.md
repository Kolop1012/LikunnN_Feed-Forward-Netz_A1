# LikunnN_Feed-Forward-Netz_A1

Ein kleines, abhängigkeitsfreies **Feed-Forward-Neuronales Netz** in reinem Java.
Das Projekt dient als Lehr- und Demonstrationsimplementierung für die Grundlagen
des überwachten Lernens: Forward-Pass, Backpropagation, Momentum und adaptive
Lernratensteuerung.

---

## Projektstruktur

```
LikunnN_Feed-Forward-Netz_A1/
├── data/                          # Trainingsdaten (CSV)
│   └── xor.csv
├── doc/                           # Ausarbeitung / Dokumentation
├── src/
│   ├── main/java/NeuralNetwork/
│   │   ├── App.java               # Konsolen-Demo / Einstiegspunkt
│   │   ├── FFNeuralNetwork.java   # Netz, Vorwärtspass, Backpropagation
│   │   ├── Layer.java             # Schicht mit Gewichten, Bias, Momentum
│   │   ├── Neuron.java            # Aktivierungswert + Aktivierungsfunktion
│   │   ├── ActivationFunction.java
│   │   ├── CostFunction.java
│   │   ├── LossFunction.java
│   │   └── LearningRateScheduler.java
│   └── test/java/NeuralNetwork/
│       └── TestFFNeuralNetwork.java
└── README.md
```

---

## Trainingsmodi

| Modus      | Momentum | Lernrate    | Besonderheit                                                                                                                            |
|------------|----------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `sgd`      | 0.0      | fest 1.0    | reines Online-SGD                                                                                                                       |
| `momentum` | 0.9      | fest 1.0    | Momentum-Term im Gewichtsupdate                                                                                                         |
| `adaptive` | 0.9      | startet 1.0 | `LearningRateScheduler` senkt die Rate, wenn sich der Fehler über 200 Epochen nicht um ≥ 0.01 verbessert (Faktor 0.5, Untergrenze 1e-5) |

Nach jeder Epoche wird der mittlere `HalfSquaredError` über alle Muster
ausgegeben. Am Ende zeigt die Demo pro Muster die Ausgabe, die
klassifizierte Klasse und das Ziel, sowie die Trefferquote.

---

## Netzarchitektur

Die demo verwendet ein Netz `2 -> 4 -> 1` mit Sigmoid-Aktivierung:

- **Eingaben:** zwei Merkmale (`x1`, `x2`).
- **Hidden-Schicht:** 4 Neuronen, Bias, Sigmoid.
- **Ausgabe:** 1 Neuron, Bias, Sigmoid; Klasse = `output ≥ 0.5`.
- **Gewichte:** Xavier-initialisiert im Bereich `±√(6 / (fanIn + fanOut))`.
- **Training:** Online-Lernen (ein Update pro Zeile), feste Reihenfolge.
- **Seed:** 42 – reproduzierbare Läufe.

Die Topologie ist in `App` fest verdrahtet, `FFNeuralNetwork` akzeptiert
aber beliebige `int[]`-Topologien.