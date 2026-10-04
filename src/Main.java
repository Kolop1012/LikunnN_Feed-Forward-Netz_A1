// Source - https://stackoverflow.com/a/2504115
// Posted by sfussenegger, modified by community. See post 'Timeline' for change history
// Retrieved 2026-10-04, License - CC BY-SA 3.0
// ja ich hatte vergessen was man für Sys.out importieren muss
import static java.lang.System.out;


public class Main {
    public static void main(String[] args) {
        FWN netz = new FWN( 3, 5, 1,  1f);
        float[] input = {1f, 1f, 0f};

        System.out.println(netz.berechneAusgabe(input));

    }
}