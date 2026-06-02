package Support;

import Support.neurone.*; // Connexion avec le package IA
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;

public class testNeurone {
	private static final float ETA_LOGIQUE = 0.1f;
	private static final float ETA_IMAGES = 0.005f;
	private static final float MSE_LIMITE_LOGIQUE = 0.001f;
	private static final float MSE_LIMITE_IMAGES = 0.05f;

	// Le pipeline est garde en memoire pour eviter de recharger le dataset
	// si plusieurs tests images sont lances dans le meme programme.
	private static DataPipeline pipelineImages = null;

	private static class ResultatEvaluation {
		double precision;
		double rsb;

		ResultatEvaluation(double precision, double rsb) {
			this.precision = precision;
			this.rsb = rsb;
		}
	}

	private static String premierRepertoireExistant(String... candidats) {
		for (String candidat : candidats) {
			Path p = Paths.get(candidat);
			if (Files.isDirectory(p)) {
				return candidat;
			}
		}
		return candidats[0];
	}

	private static void afficherMenu() {
		System.out.println();
		System.out.println("=== MENU DE TEST ===");
		System.out.println("1 - Tester le neurone sur la fonction ET");
		System.out.println("2 - Tester le neurone sur la fonction OU");
		System.out.println("3 - Tester le neurone sur la fonction XOR");
		System.out.println("4 - Tester la classification d'images sans bruit");
		System.out.println("5 - Tester la robustesse avec bruit blanc + RSB");
		System.out.println("0 - Quitter");
		System.out.print("Votre choix : ");
	}

	public static void main(String[] args) {
		Scanner clavier = new Scanner(System.in);
		boolean continuer = true;

		while (continuer) {
			afficherMenu();
			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					lancerTestAvecRepetition(clavier, () -> {
						boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
						testerFonctionLogique("ET", new float[] {0.0f, 0.0f, 0.0f, 1.0f}, afficherDetails);
					});
					break;
				case "2":
					lancerTestAvecRepetition(clavier, () -> {
						boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
						testerFonctionLogique("OU", new float[] {0.0f, 1.0f, 1.0f, 1.0f}, afficherDetails);
					});
					break;
				case "3":
					lancerTestAvecRepetition(clavier, () -> {
						boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
						testerFonctionLogique("XOR", new float[] {0.0f, 1.0f, 1.0f, 0.0f}, afficherDetails);
						System.out.println("Remarque : le XOR n'est pas lineairement separable.");
						System.out.println("Un neurone unique ne peut donc pas l'apprendre parfaitement.");
					});
					break;
				case "4":
					lancerTestAvecRepetition(clavier, () -> testerImagesSansBruit());
					break;
				case "5":
					lancerTestAvecRepetition(clavier, () -> testerRobustesseAvecBruit());
					break;
				case "0":
					continuer = false;
					System.out.println("Fin du programme de test.");
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 5.");
					break;
			}
		}

		clavier.close();
	}

	private static void lancerTestAvecRepetition(Scanner clavier, Runnable test) {
		boolean recommencer;

		do {
			test.run();
			recommencer = demanderOuiNon(clavier, "Recommencer ce test ? (o/n) : ");
		} while (recommencer);
	}

	private static boolean demanderOuiNon(Scanner clavier, String question) {
		while (true) {
			System.out.print(question);
			String reponse = clavier.nextLine().trim().toLowerCase();

			if (reponse.equals("o") || reponse.equals("oui") || reponse.equals("y") || reponse.equals("yes")) {
				return true;
			}
			if (reponse.equals("n") || reponse.equals("non") || reponse.equals("no")) {
				return false;
			}

			System.out.println("Reponse invalide. Tapez o pour oui ou n pour non.");
		}
	}

	private static void testerFonctionLogique(String nomFonction, float[] resultatsAttendus, boolean afficherDetails) {
		System.out.println();
		System.out.println("=== TEST LOGIQUE : " + nomFonction + " ===");

		final float[][] entrees = {
			{0.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f}
		};

		Neurone.fixeCoefApprentissage(ETA_LOGIQUE);

		// Utilisation de l'interface iNeurone : le type concret peut etre change
		// facilement si l'on veut comparer Heaviside, Sigmoide ou ReLU.
		iNeurone neurone = new NeuroneHeaviside(2);
		neurone.apprentissage(entrees, resultatsAttendus, MSE_LIMITE_LOGIQUE);

		int succes = 0;
		Neurone neuroneConcret = (Neurone) neurone;
		System.out.println();
		System.out.println("Entrees | Attendu | Prediction");

		if (afficherDetails) {
			System.out.println();
			System.out.println("Detail des operations :");
			System.out.println("somme = biais + entree1*w1 + entree2*w2");
			System.out.println("Heaviside : prediction = 1 si somme >= 0, sinon 0");
			System.out.println();
		}

		for (int i = 0; i < entrees.length; i++) {
			neurone.metAJour(entrees[i]);
			float prediction = (neurone.sortie() >= 0.5f) ? 1.0f : 0.0f;

			if (prediction == resultatsAttendus[i]) {
				succes++;
			}

			System.out.printf("  %.0f %.0f   |   %.0f     |    %.0f%n",
				entrees[i][0],
				entrees[i][1],
				resultatsAttendus[i],
				prediction);

			if (afficherDetails) {
				float somme = calculerSommeAvantActivation(neuroneConcret, entrees[i]);
				System.out.printf(
					"        somme = %.6f + %.0f*%.6f + %.0f*%.6f = %.6f -> Heaviside = %.0f%n",
					neuroneConcret.biais(),
					entrees[i][0],
					neuroneConcret.synapses()[0],
					entrees[i][1],
					neuroneConcret.synapses()[1],
					somme,
					prediction);
			}
		}

		double precision = (double) succes / entrees.length * 100.0;
		System.out.printf("Precision sur %s : %.2f%%%n", nomFonction, precision);

		afficherParametresNeurone(neuroneConcret);
	}

	private static float calculerSommeAvantActivation(Neurone neurone, float[] entrees) {
		float somme = neurone.biais();

		for (int i = 0; i < entrees.length; i++) {
			somme += entrees[i] * neurone.synapses()[i];
		}

		return somme;
	}

	private static DataPipeline obtenirPipelineImages() {
		if (pipelineImages != null) {
			return pipelineImages;
		}

		System.out.println();
		System.out.println("=== INITIALISATION DU PIPELINE ===");

		// Dossier dataset : compatible lancement depuis racine projet ou depuis Support/
		String repTrain = premierRepertoireExistant("dataset_groupe_7/train", "../dataset_groupe_7/train");
		String repTest = premierRepertoireExistant("dataset_groupe_7/test", "../dataset_groupe_7/test");

		pipelineImages = new DataPipeline();
		pipelineImages.construire(repTrain, repTest, true);

		return pipelineImages;
	}

	private static iNeurone entrainerNeuroneImages(DataPipeline pipeline) {
		float[][] entreesTrain = pipeline.getEntreesTrain();
		float[] labelsTrain = pipeline.getLabelsTrain();

		if (entreesTrain.length == 0) {
			System.err.println("Erreur : Donnees introuvables. Verifiez l'emplacement de 'dataset_groupe_7'.");
			return null;
		}

		int nbPixels = entreesTrain[0].length;
		System.out.printf("Nombre d'entrees du neurone (pixels) : %d synapses.%n", nbPixels);

		Neurone.fixeCoefApprentissage(ETA_IMAGES);

		// Couplage faible : on manipule le neurone via l'interface iNeurone.
		iNeurone neurone = new NeuroneHeaviside(nbPixels);

		System.out.println();
		System.out.println("=== DEBUT DE L'APPRENTISSAGE ===");
		neurone.apprentissage(entreesTrain, labelsTrain, MSE_LIMITE_IMAGES);

		return neurone;
	}

	private static void testerImagesSansBruit() {
		DataPipeline pipeline = obtenirPipelineImages();
		iNeurone neurone = entrainerNeuroneImages(pipeline);

		if (neurone == null) {
			return;
		}

		System.out.println();
		System.out.println("=== EVALUATION SUR LE FLUX DE TEST ===");

		double precision = evaluer(neurone, pipeline.getEntreesTest(), pipeline.getLabelsTest());
		System.out.printf("Precision finale sur le flux de Test : %.2f%%%n", precision);
		System.out.println("Interpretation : classification binaire chat vs non-chat.");
	}

	private static void testerRobustesseAvecBruit() {
		DataPipeline pipeline = obtenirPipelineImages();
		iNeurone neurone = entrainerNeuroneImages(pipeline);

		if (neurone == null) {
			return;
		}

		float[] amplitudes = {0.0f, 0.05f, 0.10f, 0.20f, 0.30f};

		System.out.println();
		System.out.println("=== ROBUSTESSE AU BRUIT BLANC ===");
		System.out.println("Amplitude | RSB (dB) | Precision");

		for (float amplitude : amplitudes) {
			ResultatEvaluation resultat = evaluerAvecBruit(
				neurone,
				pipeline.getEntreesTest(),
				pipeline.getLabelsTest(),
				amplitude);

			System.out.printf("   %.2f   | %7s |  %.2f%%%n",
				amplitude,
				formaterRSB(resultat.rsb),
				resultat.precision);
		}
	}

	private static double evaluer(iNeurone neurone, float[][] entrees, float[] labels) {
		int succes = 0;

		for (int i = 0; i < entrees.length; i++) {
			neurone.metAJour(entrees[i]);
			float prediction = (neurone.sortie() >= 0.5f) ? 1.0f : 0.0f;

			if (prediction == labels[i]) {
				succes++;
			}
		}

		return (double) succes / entrees.length * 100.0;
	}

	private static ResultatEvaluation evaluerAvecBruit(
		iNeurone neurone,
		float[][] entrees,
		float[] labels,
		float amplitude) {

		Random rng = new Random(1234);
		int succes = 0;
		double puissanceSignal = 0.0;
		double puissanceBruit = 0.0;
		long nbPixels = 0;

		for (int i = 0; i < entrees.length; i++) {
			float[] entreeBruitee = new float[entrees[i].length];

			for (int j = 0; j < entrees[i].length; j++) {
				float pixelOriginal = entrees[i][j];
				float bruit = (rng.nextFloat() * 2.0f - 1.0f) * amplitude;
				float pixelBruite = borner(pixelOriginal + bruit, 0.0f, 1.0f);
				float bruitReel = pixelBruite - pixelOriginal;

				entreeBruitee[j] = pixelBruite;
				puissanceSignal += pixelOriginal * pixelOriginal;
				puissanceBruit += bruitReel * bruitReel;
				nbPixels++;
			}

			neurone.metAJour(entreeBruitee);
			float prediction = (neurone.sortie() >= 0.5f) ? 1.0f : 0.0f;

			if (prediction == labels[i]) {
				succes++;
			}
		}

		double precision = (double) succes / entrees.length * 100.0;
		double rsb = calculerRSB(puissanceSignal / nbPixels, puissanceBruit / nbPixels);

		return new ResultatEvaluation(precision, rsb);
	}

	private static double calculerRSB(double puissanceSignal, double puissanceBruit) {
		if (puissanceBruit == 0.0) {
			return Double.POSITIVE_INFINITY;
		}

		return 10.0 * Math.log10(puissanceSignal / puissanceBruit);
	}

	private static String formaterRSB(double rsb) {
		if (Double.isInfinite(rsb)) {
			return "infini ";
		}

		return String.format("%.2f", rsb);
	}

	private static float borner(float valeur, float min, float max) {
		if (valeur < min) {
			return min;
		}
		if (valeur > max) {
			return max;
		}
		return valeur;
	}

	private static void afficherParametresNeurone(Neurone neurone) {
		float[] poids = neurone.synapses();

		System.out.println();
		System.out.println("Parametres appris :");
		System.out.printf("w1 = %.6f%n", poids[0]);
		System.out.printf("w2 = %.6f%n", poids[1]);
		System.out.printf("biais = %.6f%n", neurone.biais());
	}
}
