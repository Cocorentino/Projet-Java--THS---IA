package Support;

import Support.neurone.*; // Connexion avec le package IA
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
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
	private static JeuImages jeuChatChienTrain = null;
	private static JeuImages jeuChatChienTest = null;
	private static iNeurone neuroneChatChien = null;

	private static class ResultatEvaluation {
		double precision;
		double rsb;

		ResultatEvaluation(double precision, double rsb) {
			this.precision = precision;
			this.rsb = rsb;
		}
	}

	private static class JeuImages {
		float[][] entrees;
		float[] labels;
		String[] chemins;
		int nbChats;
		int nbChiens;

		JeuImages(float[][] entrees, float[] labels, String[] chemins, int nbChats, int nbChiens) {
			this.entrees = entrees;
			this.labels = labels;
			this.chemins = chemins;
			this.nbChats = nbChats;
			this.nbChiens = nbChiens;
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

	private static void afficherMenuPrincipal() {
		System.out.println();
		System.out.println("============================================================");
		System.out.println(" MENU PRINCIPAL - PROJET IA JAVA");
		System.out.println("============================================================");
		System.out.println("Etat IA chat/chien : " + (neuroneChatChien == null ? "non entrainee" : "entrainee en memoire"));
		System.out.println();
		System.out.println("1 - Validation scientifique du neurone (ET, OU, XOR)");
		System.out.println("    Sert a prouver que le neurone apprend sur des cas simples.");
		System.out.println();
		System.out.println("2 - Entrainer l'IA sur les images chat vs chien");
		System.out.println("    Charge le train, ignore wild, apprend chat=0 / chien=1.");
		System.out.println();
		System.out.println("3 - Tester l'IA entrainee");
		System.out.println("    Affiche score, fiabilite, predictions/verite, bruit + RSB.");
		System.out.println();
		System.out.println("4 - Aide et explications des choix");
		System.out.println("0 - Quitter");
		System.out.print("Votre choix : ");
	}

	public static void main(String[] args) {
		Scanner clavier = new Scanner(System.in);
		boolean continuer = true;

		while (continuer) {
			afficherMenuPrincipal();
			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					menuValidationScientifique(clavier);
					break;
				case "2":
					lancerTestAvecRepetition(clavier, () -> entrainerIAChatChien());
					break;
				case "3":
					menuTestIA(clavier);
					break;
				case "4":
					afficherAide();
					break;
				case "0":
					continuer = false;
					System.out.println("Fin du programme de test.");
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 4.");
					break;
			}
		}

		clavier.close();
	}

	private static void menuValidationScientifique(Scanner clavier) {
		boolean retour = false;

		while (!retour) {
			System.out.println();
			System.out.println("------------------------------------------------------------");
			System.out.println(" VALIDATION SCIENTIFIQUE DU NEURONE");
			System.out.println("------------------------------------------------------------");
			System.out.println("Ces tests ne reconnaissent pas encore les images.");
			System.out.println("Ils servent a verifier que l'apprentissage fonctionne sur");
			System.out.println("des fonctions simples, comme demande dans la demarche prof.");
			System.out.println();
			System.out.println("1 - Fonction ET : doit sortir 1 seulement pour 1 ET 1");
			System.out.println("2 - Fonction OU : doit sortir 1 si au moins une entree vaut 1");
			System.out.println("3 - Fonction XOR : test limite, impossible pour un seul neurone");
			System.out.println("4 - Lancer ET puis OU puis XOR");
			System.out.println("0 - Retour au menu principal");
			System.out.print("Votre choix : ");

			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					lancerTestLogiqueAvecOptions(clavier, "ET", new float[] {0.0f, 0.0f, 0.0f, 1.0f});
					break;
				case "2":
					lancerTestLogiqueAvecOptions(clavier, "OU", new float[] {0.0f, 1.0f, 1.0f, 1.0f});
					break;
				case "3":
					lancerTestLogiqueAvecOptions(clavier, "XOR", new float[] {0.0f, 1.0f, 1.0f, 0.0f});
					System.out.println("Remarque : le XOR n'est pas lineairement separable.");
					System.out.println("Un neurone unique ne peut donc pas l'apprendre parfaitement.");
					break;
				case "4":
					boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
					testerFonctionLogique("ET", new float[] {0.0f, 0.0f, 0.0f, 1.0f}, afficherDetails);
					testerFonctionLogique("OU", new float[] {0.0f, 1.0f, 1.0f, 1.0f}, afficherDetails);
					testerFonctionLogique("XOR", new float[] {0.0f, 1.0f, 1.0f, 0.0f}, afficherDetails);
					System.out.println("Remarque : le XOR illustre volontairement la limite du perceptron simple.");
					break;
				case "0":
					retour = true;
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 4.");
					break;
			}
		}
	}

	private static void menuTestIA(Scanner clavier) {
		boolean retour = false;

		while (!retour) {
			System.out.println();
			System.out.println("------------------------------------------------------------");
			System.out.println(" TESTER L'IA IMAGE");
			System.out.println("------------------------------------------------------------");
			System.out.println("Etat IA chat/chien : " + (neuroneChatChien == null ? "non entrainee" : "entrainee en memoire"));
			System.out.println();
			System.out.println("1 - Tester chat vs chien avec score global");
			System.out.println("2 - Tester chat vs chien avec prediction/verite image par image");
			System.out.println("3 - Tester la robustesse chat vs chien avec bruit blanc + RSB");
			System.out.println("4 - Ancien test : chat vs non-chat sans bruit");
			System.out.println("5 - Ancien test : chat vs non-chat avec bruit blanc + RSB");
			System.out.println("0 - Retour au menu principal");
			System.out.print("Votre choix : ");

			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					lancerTestAvecRepetition(clavier, () -> testerIAChatChien(false, 0));
					break;
				case "2":
					lancerTestAvecRepetition(clavier, () -> {
						int limiteAffichage = demanderEntier(
							clavier,
							"Nombre d'images a afficher (0 = toutes) : ",
							0);
						testerIAChatChien(true, limiteAffichage);
					});
					break;
				case "3":
					lancerTestAvecRepetition(clavier, () -> testerRobustesseChatChien());
					break;
				case "4":
					lancerTestAvecRepetition(clavier, () -> testerImagesSansBruit());
					break;
				case "5":
					lancerTestAvecRepetition(clavier, () -> testerRobustesseAvecBruit());
					break;
				case "0":
					retour = true;
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 5.");
					break;
			}
		}
	}

	private static void lancerTestLogiqueAvecOptions(Scanner clavier, String nomFonction, float[] resultatsAttendus) {
		lancerTestAvecRepetition(clavier, () -> {
			boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
			testerFonctionLogique(nomFonction, resultatsAttendus, afficherDetails);
		});
	}

	private static void afficherAide() {
		System.out.println();
		System.out.println("============================================================");
		System.out.println(" AIDE - A QUOI SERVENT LES MENUS ?");
		System.out.println("============================================================");
		System.out.println("1. Validation scientifique");
		System.out.println("   ET et OU ne servent pas a reconnaitre les images.");
		System.out.println("   Ils prouvent que le neurone apprend correctement sur un");
		System.out.println("   probleme simple et lineairement separable.");
		System.out.println("   XOR sert a montrer une limite normale : un seul neurone");
		System.out.println("   ne peut pas tout apprendre.");
		System.out.println();
		System.out.println("2. Entrainer l'IA");
		System.out.println("   Le programme lit les images train, garde cat/dog, normalise");
		System.out.println("   les pixels, melange le train, puis ajuste les poids du neurone.");
		System.out.println("   Le neurone entraine reste en memoire tant que le programme tourne.");
		System.out.println();
		System.out.println("3. Tester l'IA");
		System.out.println("   Le programme applique le neurone entraine aux images de test.");
		System.out.println("   Il compare ce que l'IA trouve avec la verite issue du dossier.");
		System.out.println("   Le score final donne le pourcentage de fiabilite.");
		System.out.println();
		System.out.println("4. Bruit blanc + RSB");
		System.out.println("   On degrade volontairement les pixels avec du bruit controle.");
		System.out.println("   Le RSB mesure le rapport signal/bruit : plus il baisse, plus");
		System.out.println("   le test est difficile pour le neurone.");
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

	private static int demanderEntier(Scanner clavier, String question, int minimum) {
		while (true) {
			System.out.print(question);
			String reponse = clavier.nextLine().trim();

			try {
				int valeur = Integer.parseInt(reponse);
				if (valeur >= minimum) {
					return valeur;
				}
			} catch (NumberFormatException e) {
				// La reponse sera traitee par le message d'erreur commun ci-dessous.
			}

			System.out.printf("Reponse invalide. Entrez un entier superieur ou egal a %d.%n", minimum);
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

	private static void entrainerIAChatChien() {
		System.out.println();
		System.out.println("=== ENTRAINEMENT IA CHAT VS CHIEN ===");
		System.out.println("Convention : chat = 0, chien = 1.");
		System.out.println("Les images wild sont ignorees pour cet apprentissage binaire.");

		JeuImages train = obtenirJeuChatChienTrain();

		if (train.entrees.length == 0) {
			System.err.println("Erreur : donnees d'entrainement chat/chien introuvables.");
			return;
		}

		System.out.printf(
			"Train chat/chien : %d images (%d chats, %d chiens).%n",
			train.entrees.length,
			train.nbChats,
			train.nbChiens);

		neuroneChatChien = entrainerNeuroneChatChien(train);
		System.out.println("IA chat/chien entrainee et gardee en memoire.");
	}

	private static void testerIAChatChien(boolean afficherDetails, int limiteAffichage) {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages test = obtenirJeuChatChienTest();

		if (test.entrees.length == 0) {
			System.err.println("Erreur : donnees de test chat/chien introuvables.");
			return;
		}

		System.out.println();
		System.out.println("=== TEST IA CHAT VS CHIEN ===");
		System.out.printf(
			"Test chat/chien : %d images (%d chats, %d chiens).%n",
			test.entrees.length,
			test.nbChats,
			test.nbChiens);

		evaluerChatVsChien(neuroneChatChien, test, afficherDetails, limiteAffichage);
	}

	private static void testerRobustesseChatChien() {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages test = obtenirJeuChatChienTest();
		float[] amplitudes = {0.0f, 0.05f, 0.10f, 0.20f, 0.30f};

		System.out.println();
		System.out.println("=== ROBUSTESSE CHAT VS CHIEN AU BRUIT BLANC ===");
		System.out.println("Amplitude | RSB (dB) | Fiabilite");

		for (float amplitude : amplitudes) {
			ResultatEvaluation resultat = evaluerAvecBruit(
				neuroneChatChien,
				test.entrees,
				test.labels,
				amplitude);

			System.out.printf("   %.2f   | %7s |  %.2f%%%n",
				amplitude,
				formaterRSB(resultat.rsb),
				resultat.precision);
		}
	}

	private static boolean iaChatChienPrete() {
		if (neuroneChatChien != null) {
			return true;
		}

		System.out.println();
		System.out.println("IA chat/chien non entrainee.");
		System.out.println("Choisissez d'abord l'option 2 du menu principal pour l'entrainer.");
		return false;
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

	private static JeuImages obtenirJeuChatChienTrain() {
		if (jeuChatChienTrain == null) {
			String repTrain = premierRepertoireExistant("dataset_groupe_7/train", "../dataset_groupe_7/train");
			jeuChatChienTrain = chargerJeuChatChien(repTrain, true, true);
		}

		return jeuChatChienTrain;
	}

	private static JeuImages obtenirJeuChatChienTest() {
		if (jeuChatChienTest == null) {
			String repTest = premierRepertoireExistant("dataset_groupe_7/test", "../dataset_groupe_7/test");
			jeuChatChienTest = chargerJeuChatChien(repTest, true, false);
		}

		return jeuChatChienTest;
	}

	private static JeuImages chargerJeuChatChien(String repertoire, boolean niveauxDeGris, boolean melanger) {
		List<String> chemins = Image.listeFichiers(repertoire);
		List<float[]> entrees = new ArrayList<>();
		List<Float> labels = new ArrayList<>();
		List<String> cheminsConserves = new ArrayList<>();
		int nbChats = 0;
		int nbChiens = 0;

		if (chemins == null) {
			System.err.println("Erreur : Impossible de lire le repertoire " + repertoire);
			return new JeuImages(new float[0][], new float[0], new String[0], 0, 0);
		}

		for (String chemin : chemins) {
			String lower = chemin.toLowerCase();
			if (!lower.endsWith(".jpg") && !lower.endsWith(".jpeg") && !lower.endsWith(".png")) {
				continue;
			}

			int labelOriginal = DataPipeline.extraireLabel(chemin);
			if (labelOriginal != 0 && labelOriginal != 1) {
				continue;
			}

			Image image = new Image(chemin, labelOriginal, niveauxDeGris);
			if (image.donnees() == null) {
				continue;
			}

			entrees.add(DataPipeline.normaliser(image.donnees()));
			labels.add(labelOriginal == 1 ? 1.0f : 0.0f);
			cheminsConserves.add(chemin);

			if (labelOriginal == 0) {
				nbChats++;
			} else {
				nbChiens++;
			}
		}

		JeuImages jeu = convertirEnTableaux(entrees, labels, cheminsConserves, nbChats, nbChiens);

		if (melanger) {
			melangerJeuImages(jeu);
		}

		return jeu;
	}

	private static JeuImages convertirEnTableaux(
		List<float[]> entrees,
		List<Float> labels,
		List<String> chemins,
		int nbChats,
		int nbChiens) {

		float[][] entreesTableau = new float[entrees.size()][];
		float[] labelsTableau = new float[labels.size()];
		String[] cheminsTableau = new String[chemins.size()];

		for (int i = 0; i < entrees.size(); i++) {
			entreesTableau[i] = entrees.get(i);
			labelsTableau[i] = labels.get(i);
			cheminsTableau[i] = chemins.get(i);
		}

		return new JeuImages(entreesTableau, labelsTableau, cheminsTableau, nbChats, nbChiens);
	}

	private static void melangerJeuImages(JeuImages jeu) {
		Random rng = new Random();

		for (int i = jeu.entrees.length - 1; i > 0; i--) {
			int j = rng.nextInt(i + 1);

			float[] tmpEntree = jeu.entrees[i];
			jeu.entrees[i] = jeu.entrees[j];
			jeu.entrees[j] = tmpEntree;

			float tmpLabel = jeu.labels[i];
			jeu.labels[i] = jeu.labels[j];
			jeu.labels[j] = tmpLabel;

			String tmpChemin = jeu.chemins[i];
			jeu.chemins[i] = jeu.chemins[j];
			jeu.chemins[j] = tmpChemin;
		}
	}

	private static iNeurone entrainerNeuroneChatChien(JeuImages train) {
		int nbPixels = train.entrees[0].length;
		System.out.printf("Nombre d'entrees du neurone (pixels) : %d synapses.%n", nbPixels);

		Neurone.fixeCoefApprentissage(ETA_IMAGES);

		// Couplage faible : le test reste compatible avec Heaviside, Sigmoide ou ReLU.
		iNeurone neurone = new NeuroneHeaviside(nbPixels);

		System.out.println();
		System.out.println("=== APPRENTISSAGE CHAT VS CHIEN ===");
		neurone.apprentissage(train.entrees, train.labels, MSE_LIMITE_IMAGES);

		return neurone;
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

	private static void evaluerChatVsChien(
		iNeurone neurone,
		JeuImages test,
		boolean afficherDetails,
		int limiteAffichage) {

		int succes = 0;
		int nbAffiches = 0;

		System.out.println();
		System.out.println("=== RESULTATS CHAT VS CHIEN ===");

		if (afficherDetails) {
			System.out.println("No | IA trouve | Verite | Resultat | Fichier");
		}

		for (int i = 0; i < test.entrees.length; i++) {
			neurone.metAJour(test.entrees[i]);
			float prediction = (neurone.sortie() >= 0.5f) ? 1.0f : 0.0f;
			boolean correct = prediction == test.labels[i];

			if (correct) {
				succes++;
			}

			if (afficherDetails && (limiteAffichage == 0 || nbAffiches < limiteAffichage)) {
				System.out.printf(
					"%4d | %-8s | %-6s | %-7s | %s%n",
					i + 1,
					nomClasseChatChien(prediction),
					nomClasseChatChien(test.labels[i]),
					correct ? "OK" : "ERREUR",
					nomFichier(test.chemins[i]));
				nbAffiches++;
			}
		}

		double fiabilite = (double) succes / test.entrees.length * 100.0;
		System.out.printf("Score : %d/%d bonnes predictions.%n", succes, test.entrees.length);
		System.out.printf("Pourcentage de fiabilite : %.2f%%%n", fiabilite);

		if (afficherDetails && limiteAffichage > 0 && limiteAffichage < test.entrees.length) {
			System.out.printf(
				"Affichage limite a %d images sur %d pour garder le terminal lisible.%n",
				limiteAffichage,
				test.entrees.length);
		}
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

	private static String nomClasseChatChien(float label) {
		return label >= 0.5f ? "chien" : "chat";
	}

	private static String nomFichier(String chemin) {
		return Paths.get(chemin).getFileName().toString();
	}
}
