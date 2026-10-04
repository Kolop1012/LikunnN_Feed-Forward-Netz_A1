public class FWN {

    private float[] eingabeGewichte;
    private float[][] gewichteEingabeNeuronen;
    private float[] gewichteVerborgeneSchicht;
    private float ausgabeGewicht;

    private float maximalesGewicht;


    private float [] EingagbeNeuronen = new float[3];
    private float [] NeuronenVerborgeneSchicht;
    private float AusgabeNeuron = 1f;


    public FWN(int anzahlEingabeneuronen, int anzahlNeuronenVerborgeneSchicht, int anzahlAusgabeNeuronen, float maximalesGewicht){
        this.maximalesGewicht = maximalesGewicht;

        NeuronenVerborgeneSchicht = new float[anzahlNeuronenVerborgeneSchicht];


        eingabeGewichte = new float[anzahlEingabeneuronen];
        initGewichte(eingabeGewichte, false);


        gewichteEingabeNeuronen = new float[anzahlEingabeneuronen][anzahlNeuronenVerborgeneSchicht];
        for(int i = 0; i < anzahlEingabeneuronen; i++){
            initGewichte(gewichteEingabeNeuronen[i], true);
        }


        gewichteVerborgeneSchicht = new float[anzahlNeuronenVerborgeneSchicht];
        initGewichte(gewichteVerborgeneSchicht, true);

    }

    public void initGewichte(float[] gewichte, boolean zufaelligeGewichte){
        for(int i = 0; i < gewichte.length; i++){
            if(zufaelligeGewichte){
                gewichte[i] = (0 + (float)(Math.random()) * (maximalesGewicht - 0));
            }
            else{
                gewichte[i] = maximalesGewicht;
            }
        }
    }


    public float berechneAusgabe(float[] input){
        for(int i = 0; i<EingagbeNeuronen.length; i++){
            EingagbeNeuronen[i] = input[i]*eingabeGewichte[i];
        }


        for(int i = 0; i<NeuronenVerborgeneSchicht.length; i++){
            for(int j = 0; j<EingagbeNeuronen.length; j++){
                NeuronenVerborgeneSchicht[i] += EingagbeNeuronen[j]*gewichteEingabeNeuronen[j][i];
            }

        }

        for(int i = 0; i < NeuronenVerborgeneSchicht.length; i++){
            AusgabeNeuron += NeuronenVerborgeneSchicht[i]*gewichteVerborgeneSchicht[i];
        }
        return AusgabeNeuron;
    }

}
