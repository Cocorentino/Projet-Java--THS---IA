package Support;

import java.util.*;

/**
 * Pôle 2 — Data Pipeline & Prétraitement
 * Gère le chargement, la normalisation, le mélange et la séparation
 * train/test des images pour l'apprentissage du neurone.
 */
public class DataPipeline {
 
    // =========================================================
    // TÂCHE 2.4 — Structure de données globale train / test
    // =========================================================
    private float[][] entreesTrain;
    private float[] labelsTrain;
    private float[][] entreesTest;
    private float[] labelsTest;
 
    // Accesseurs
    public float[][] getEntreesTrain() { return entreesTrain; }
    public float[]   getLabelsTrain()  { return labelsTrain;  }
    public float[][] getEntreesTest()  { return entreesTest;  }
    public float[]   getLabelsTest()   { return labelsTest;   }
 
    // =========================================================
    // TÂCHE 2.1 — Extraction et labellisation
    // =========================================================
    /**
     * Détermine le label numérique d'une image à partir de son chemin.
     * 0 pour chat, 1 pour chien, 2 pour sauvage, 3 inconnu.
     */
    public static int extraireLabel(String chemin) {
        String c = chemin.replace("\\", "/").toLowerCase();
 
        if (c.contains("/cat/")  || c.contains("/cats/"))  return 0; // Chat = 0
        if (c.contains("/dog/")  || c.contains("/dogs/"))  return 1; // Chien = 1
        if (c.contains("/wild/") || c.contains("/wilds/")) return 2; // Wild = 2
        return 3; // Inconnu
    }
 
    /**
     * Charge toutes les images d'un répertoire via Image.listeFichiers()
     */
    public static List<Image> chargerImages(String repertoire, boolean niveauxDeGris) {
        List<String> chemins = Image.listeFichiers(repertoire);
        List<Image> images = new ArrayList<>();
 
        if (chemins == null) {
            System.err.println("Erreur : Impossible de lire le répertoire " + repertoire);
            return images;
        }
 
        for (String chemin : chemins) {
            String lower = chemin.toLowerCase();
            if (!lower.endsWith(".jpg") && !lower.endsWith(".jpeg") && !lower.endsWith(".png")) {
                continue;
            }
 
            int label = extraireLabel(chemin);
            if (label == 3) {
                continue; 
            }
 
            images.add(new Image(chemin, label, niveauxDeGris));
        }
 
        System.out.printf("Chargement : %d images valides trouvees dans '%s'%n", images.size(), repertoire);
        return images;
    }
 
    // =========================================================
    // TÂCHE 2.2 — Algorithme de Normalisation (0-255 -> 0.0-1.0)
    // =========================================================
    public static float[] normaliser(int[] pixels) {
        float[] normalises = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            normalises[i] = pixels[i] / 255.0f;  
        }
        return normalises;
    }
 
    // =========================================================
    // TÂCHE 2.3 — Algorithme de Mélange (Fisher-Yates)
    // =========================================================
    public static void melangerDonnees(float[][] entrees, float[] labels) {
        Random rng = new Random();
        int n = entrees.length;
 
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1); 
 
            // Échange des matrices de pixels
            float[] tmpEntree = entrees[i];
            entrees[i] = entrees[j];
            entrees[j] = tmpEntree;
 
            // Échange des labels en parallèle
            float tmpLabel = labels[i];
            labels[i] = labels[j];
            labels[j] = tmpLabel;
        }
    }
 
    // =========================================================
    // TÂCHE 2.4 — Flux de Train et Flux de Test
    // =========================================================
    public void construire(String repTrain, String repTest, boolean niveauxDeGris) {
 
        // 1. Flux d'entraînement
        List<Image> imagesTrain = chargerImages(repTrain, niveauxDeGris);
        entreesTrain = new float[imagesTrain.size()][];
        labelsTrain  = new float[imagesTrain.size()];
 
        for (int i = 0; i < imagesTrain.size(); i++) {
            Image img = imagesTrain.get(i);
            entreesTrain[i] = normaliser(img.donnees());  
            // Actif (1.0f) = chat (label 0), Inactif (0.0f) = autre
            labelsTrain[i] = (img.label() == 0) ? 1.0f : 0.0f;
        }
 
        // Mélange obligatoire du train
        melangerDonnees(entreesTrain, labelsTrain);
 
        // 2. Flux de test (Pas de mélange ici)
        List<Image> imagesTest = chargerImages(repTest, niveauxDeGris);
        entreesTest = new float[imagesTest.size()][];
        labelsTest  = new float[imagesTest.size()];
 
        for (int i = 0; i < imagesTest.size(); i++) {
            Image img = imagesTest.get(i);
            entreesTest[i] = normaliser(img.donnees());
            labelsTest[i]  = (img.label() == 0) ? 1.0f : 0.0f;
        }
 
        System.out.printf("Pipeline pret : %d images de train, %d images de test.%n", entreesTrain.length, entreesTest.length);
    }
}
