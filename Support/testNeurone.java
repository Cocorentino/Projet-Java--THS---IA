package Support;

import Support.neurone.*; // Connexion avec le package IA
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;

public class testNeurone {
	private static final float ETA_LOGIQUE = 0.1f;
	private static final float ETA_IMAGES_DEFAUT = 0.005f;
	private static final float MSE_LIMITE_LOGIQUE = 0.001f;
	private static final float MSE_LIMITE_IMAGES_DEFAUT = 0.05f;
	private static final int EPOCHS_IMAGES_DEFAUT = 500;
	private static final String DOSSIER_MODELES = "modeles";
	private static final String FICHIER_POIDS_MODELE = DOSSIER_MODELES + "/meilleur_chat_chien.txt";
	private static final String FICHIER_META_MODELE = DOSSIER_MODELES + "/meilleur_chat_chien.meta";

	// Les jeux d'images sont gardes en memoire pour eviter de relire le dataset
	// si plusieurs tests sont lances dans la meme execution du programme.
	private static DataPipeline pipelineImagesGris = null;
	private static DataPipeline pipelineImagesRGB = null;
	private static JeuImages jeuChatChienTrainGris = null;
	private static JeuImages jeuChatChienTestGris = null;
	private static JeuImages jeuChatChienTrainRGB = null;
	private static JeuImages jeuChatChienTestRGB = null;
	private static ModeleIA modeleChatChien = null;

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
		boolean niveauxDeGris;

		JeuImages(
			float[][] entrees,
			float[] labels,
			String[] chemins,
			int nbChats,
			int nbChiens,
			boolean niveauxDeGris) {

			this.entrees = entrees;
			this.labels = labels;
			this.chemins = chemins;
			this.nbChats = nbChats;
			this.nbChiens = nbChiens;
			this.niveauxDeGris = niveauxDeGris;
		}
	}

	private static class ModeleIA {
		iNeurone neurone;
		String activation;
		float eta;
		int epochsMax;
		float mseLimite;
		double precisionValidation;
		double precisionTest;
		int nbPixels;
		boolean niveauxDeGris;

		ModeleIA(
			iNeurone neurone,
			String activation,
			float eta,
			int epochsMax,
			float mseLimite,
			double precisionValidation,
			double precisionTest,
			int nbPixels,
			boolean niveauxDeGris) {

			this.neurone = neurone;
			this.activation = activation;
			this.eta = eta;
			this.epochsMax = epochsMax;
			this.mseLimite = mseLimite;
			this.precisionValidation = precisionValidation;
			this.precisionTest = precisionTest;
			this.nbPixels = nbPixels;
			this.niveauxDeGris = niveauxDeGris;
		}
	}

	public static void main(String[] args) {
		Scanner clavier = new Scanner(System.in);

		// Si un modele a deja ete sauvegarde, on le recharge automatiquement.
		// Cela evite de devoir reentrainer a chaque lancement du programme.
		chargerModeleSauvegarde(false);

		boolean continuer = true;
		while (continuer) {
			afficherMenuPrincipal();
			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					menuValidationScientifique(clavier);
					break;
				case "2":
					menuEntrainerOptimiser(clavier);
					break;
				case "3":
					menuTesterIA(clavier);
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

	private static void afficherMenuPrincipal() {
		System.out.println();
		System.out.println("============================================================");
		System.out.println(" MENU PRINCIPAL - PROJET IA JAVA");
		System.out.println("============================================================");
		System.out.println("Etat IA chat/chien : " + etatModeleCourant());
		System.out.println();
		System.out.println("1 - Validation scientifique du neurone");
		System.out.println("    ET / OU / XOR pour prouver que le neurone apprend.");
		System.out.println();
		System.out.println("2 - Entrainer et optimiser l'IA chat vs chien");
		System.out.println("    Choix activation, eta, epochs, benchmark et sauvegarde.");
		System.out.println();
		System.out.println("3 - Tester l'IA entrainee");
		System.out.println("    Score, prediction/verite, train vs test, bruit blanc + RSB.");
		System.out.println();
		System.out.println("4 - Aide et explications des choix");
		System.out.println("0 - Quitter");
		System.out.print("Votre choix : ");
	}

	private static String etatModeleCourant() {
		if (modeleChatChien == null) {
			return "non entrainee";
		}

		return String.format(
			"entrainee (%s, %s, eta=%.5f, epochs=%d, test=%.2f%%)",
			modeleChatChien.activation,
			nomRepresentation(modeleChatChien.niveauxDeGris),
			modeleChatChien.eta,
			modeleChatChien.epochsMax,
			modeleChatChien.precisionTest);
	}

	private static void menuValidationScientifique(Scanner clavier) {
		boolean retour = false;

		while (!retour) {
			System.out.println();
			System.out.println("------------------------------------------------------------");
			System.out.println(" VALIDATION SCIENTIFIQUE DU NEURONE");
			System.out.println("------------------------------------------------------------");
			System.out.println("Ces tests ne reconnaissent pas les images directement.");
			System.out.println("Ils valident la base mathematique de l'apprentissage.");
			System.out.println();
			System.out.println("1 - Fonction ET : sortie 1 seulement pour 1 ET 1");
			System.out.println("2 - Fonction OU : sortie 1 si au moins une entree vaut 1");
			System.out.println("3 - Fonction XOR : limite normale d'un neurone unique");
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
					System.out.println("Remarque : XOR illustre volontairement la limite du perceptron simple.");
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

	private static void menuEntrainerOptimiser(Scanner clavier) {
		boolean retour = false;

		while (!retour) {
			System.out.println();
			System.out.println("------------------------------------------------------------");
			System.out.println(" ENTRAINER ET OPTIMISER L'IA CHAT VS CHIEN");
			System.out.println("------------------------------------------------------------");
			System.out.println("Etat courant : " + etatModeleCourant());
			System.out.println();
			System.out.println("1 - Entrainement rapide en niveaux de gris");
			System.out.println("2 - Entrainement rapide en RGB");
			System.out.println("3 - Entrainement personnalise");
			System.out.println("4 - Continuer l'entrainement du modele courant");
			System.out.println("5 - Benchmark automatique et garder le meilleur modele");
			System.out.println("6 - Sauvegarder le modele courant");
			System.out.println("7 - Charger le modele sauvegarde");
			System.out.println("0 - Retour au menu principal");
			System.out.print("Votre choix : ");

			String choix = clavier.nextLine().trim();

			switch (choix) {
				case "1":
					entrainerIAChatChien(
						"Heaviside",
						ETA_IMAGES_DEFAUT,
						EPOCHS_IMAGES_DEFAUT,
						MSE_LIMITE_IMAGES_DEFAUT,
						true);
					break;
				case "2":
					entrainerIAChatChien(
						"Heaviside",
						ETA_IMAGES_DEFAUT,
						EPOCHS_IMAGES_DEFAUT,
						MSE_LIMITE_IMAGES_DEFAUT,
						false);
					break;
				case "3":
					entrainerIAChatChienPersonnalise(clavier);
					break;
				case "4":
					continuerEntrainementModeleCourant(clavier);
					break;
				case "5":
					executerBenchmark(clavier);
					break;
				case "6":
					sauvegarderModeleCourant();
					break;
				case "7":
					chargerModeleSauvegarde(true);
					break;
				case "0":
					retour = true;
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 7.");
					break;
			}
		}
	}

	private static void menuTesterIA(Scanner clavier) {
		boolean retour = false;

		while (!retour) {
			System.out.println();
			System.out.println("------------------------------------------------------------");
			System.out.println(" TESTER L'IA IMAGE");
			System.out.println("------------------------------------------------------------");
			System.out.println("Etat courant : " + etatModeleCourant());
			System.out.println();
			System.out.println("1 - Tester chat vs chien avec score global");
			System.out.println("2 - Tester chat vs chien avec prediction/verite image par image");
			System.out.println("3 - Comparer score train et score test");
			System.out.println("4 - Tester la robustesse chat vs chien avec bruit blanc + RSB");
			System.out.println("5 - Ancien test : chat vs non-chat sans bruit");
			System.out.println("6 - Ancien test : chat vs non-chat avec bruit blanc + RSB");
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
					lancerTestAvecRepetition(clavier, () -> comparerTrainEtTest());
					break;
				case "4":
					lancerTestAvecRepetition(clavier, () -> testerRobustesseChatChien());
					break;
				case "5":
					lancerTestAvecRepetition(clavier, () -> {
						boolean niveauxDeGris = demanderRepresentationImages(clavier, true);
						testerImagesSansBruit(niveauxDeGris);
					});
					break;
				case "6":
					lancerTestAvecRepetition(clavier, () -> {
						boolean niveauxDeGris = demanderRepresentationImages(clavier, true);
						testerRobustesseAvecBruit(niveauxDeGris);
					});
					break;
				case "0":
					retour = true;
					break;
				default:
					System.out.println("Choix invalide. Entrez un nombre entre 0 et 6.");
					break;
			}
		}
	}

	private static void afficherAide() {
		System.out.println();
		System.out.println("============================================================");
		System.out.println(" AIDE - COMMENT OBTENIR UN BON SCORE SANS SORTIR DU SUJET");
		System.out.println("============================================================");
		System.out.println("1. ET / OU / XOR");
		System.out.println("   Ces tests ne classent pas les images. Ils prouvent que le");
		System.out.println("   neurone et son apprentissage fonctionnent sur des cas simples.");
		System.out.println();
		System.out.println("2. Entrainement personnalise");
		System.out.println("   Activation : Heaviside, Sigmoide ou ReLU.");
		System.out.println("   Representation : niveaux de gris ou RGB.");
		System.out.println("   RGB garde les canaux rouge, vert et bleu, donc 3 fois plus");
		System.out.println("   d'entrees que le gris. C'est plus lent et pas forcement meilleur.");
		System.out.println("   Eta : vitesse de correction des poids.");
		System.out.println("   Epochs : nombre maximal de passages sur le train.");
		System.out.println("   MSE limite : seuil d'erreur vise pendant l'apprentissage.");
		System.out.println();
		System.out.println("3. Benchmark");
		System.out.println("   Le programme teste plusieurs configurations et garde celle");
		System.out.println("   qui obtient le meilleur score sur une partie validation.");
		System.out.println("   Le score officiel reste celui sur le dossier test.");
		System.out.println();
		System.out.println("4. Surapprentissage");
		System.out.println("   Un score train tres haut avec un score test faible signifie");
		System.out.println("   que le modele memorise trop le train et generalise mal.");
		System.out.println();
		System.out.println("5. Bruit blanc + RSB");
		System.out.println("   On degrade les pixels avec un bruit controle pour mesurer");
		System.out.println("   la robustesse du neurone.");
	}

	private static void lancerTestLogiqueAvecOptions(Scanner clavier, String nomFonction, float[] resultatsAttendus) {
		lancerTestAvecRepetition(clavier, () -> {
			boolean afficherDetails = demanderOuiNon(clavier, "Afficher le detail des operations ? (o/n) : ");
			testerFonctionLogique(nomFonction, resultatsAttendus, afficherDetails);
		});
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

	private static String demanderTexteAvecDefaut(Scanner clavier, String question, String valeurDefaut) {
		System.out.print(question + " [" + valeurDefaut + "] : ");
		String reponse = clavier.nextLine().trim();
		return reponse.isEmpty() ? valeurDefaut : reponse;
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

	private static int demanderEntierAvecDefaut(Scanner clavier, String question, int minimum, int valeurDefaut) {
		while (true) {
			String texte = demanderTexteAvecDefaut(clavier, question, String.valueOf(valeurDefaut));

			try {
				int valeur = Integer.parseInt(texte);
				if (valeur >= minimum) {
					return valeur;
				}
			} catch (NumberFormatException e) {
				// La reponse sera traitee par le message d'erreur commun ci-dessous.
			}

			System.out.printf("Reponse invalide. Entrez un entier superieur ou egal a %d.%n", minimum);
		}
	}

	private static float demanderFloatAvecDefaut(Scanner clavier, String question, float minimum, float valeurDefaut) {
		while (true) {
			String texte = demanderTexteAvecDefaut(clavier, question, String.valueOf(valeurDefaut));

			try {
				float valeur = Float.parseFloat(texte.replace(",", "."));
				if (valeur >= minimum) {
					return valeur;
				}
			} catch (NumberFormatException e) {
				// La reponse sera traitee par le message d'erreur commun ci-dessous.
			}

			System.out.printf("Reponse invalide. Entrez un nombre superieur ou egal a %.6f.%n", minimum);
		}
	}

	private static String demanderActivation(Scanner clavier, String valeurDefaut) {
		while (true) {
			System.out.println();
			System.out.println("Fonction d'activation :");
			System.out.println("1 - Heaviside : sortie 0 ou 1, perceptron classique");
			System.out.println("2 - Sigmoide  : sortie progressive entre 0 et 1");
			System.out.println("3 - ReLU      : sortie 0 si negatif, sinon valeur positive");
			String choix = demanderTexteAvecDefaut(clavier, "Votre choix", valeurDefaut);

			if (choix.equals("1") || choix.equalsIgnoreCase("Heaviside")) {
				return "Heaviside";
			}
			if (choix.equals("2") || choix.equalsIgnoreCase("Sigmoide") || choix.equalsIgnoreCase("Sigmoid")) {
				return "Sigmoide";
			}
			if (choix.equals("3") || choix.equalsIgnoreCase("ReLU")) {
				return "ReLU";
			}

			System.out.println("Activation inconnue.");
		}
	}

	private static boolean demanderRepresentationImages(Scanner clavier, boolean defautNiveauxDeGris) {
		String valeurDefaut = defautNiveauxDeGris ? "1" : "2";

		while (true) {
			System.out.println();
			System.out.println("Representation des images :");
			System.out.println("1 - Niveaux de gris : 1 valeur par pixel, apprentissage plus rapide");
			System.out.println("2 - RGB             : 3 valeurs par pixel, conserve la couleur");
			String choix = demanderTexteAvecDefaut(clavier, "Votre choix", valeurDefaut);

			if (choix.equals("1") || choix.equalsIgnoreCase("gris") || choix.equalsIgnoreCase("gray")) {
				return true;
			}
			if (choix.equals("2") || choix.equalsIgnoreCase("rgb") || choix.equalsIgnoreCase("couleur")) {
				return false;
			}

			System.out.println("Representation inconnue. Entrez 1 pour gris ou 2 pour RGB.");
		}
	}

	private static String nomRepresentation(boolean niveauxDeGris) {
		return niveauxDeGris ? "niveaux de gris" : "RGB";
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
		Neurone.fixeNbEpochsMax(500);
		Neurone.fixeTraceApprentissage(true);

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
			float prediction = predictionBinaire(neurone);

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

	private static void entrainerIAChatChienPersonnalise(Scanner clavier) {
		String activation = demanderActivation(clavier, "1");
		boolean niveauxDeGris = demanderRepresentationImages(clavier, true);
		float eta = demanderFloatAvecDefaut(clavier, "Coefficient d'apprentissage eta", 0.000001f, ETA_IMAGES_DEFAUT);
		int epochsMax = demanderEntierAvecDefaut(clavier, "Nombre maximal d'epochs", 1, EPOCHS_IMAGES_DEFAUT);
		float mseLimite = demanderFloatAvecDefaut(clavier, "MSE limite", 0.0f, MSE_LIMITE_IMAGES_DEFAUT);

		entrainerIAChatChien(activation, eta, epochsMax, mseLimite, niveauxDeGris);
	}

	private static void entrainerIAChatChien(
		String activation,
		float eta,
		int epochsMax,
		float mseLimite,
		boolean niveauxDeGris) {

		System.out.println();
		System.out.println("=== ENTRAINEMENT IA CHAT VS CHIEN ===");
		System.out.println("Convention : chat = 0, chien = 1.");
		System.out.println("Les images wild sont ignorees pour cet apprentissage binaire.");
		System.out.println("Representation : " + nomRepresentation(niveauxDeGris) + ".");

		JeuImages train = obtenirJeuChatChienTrain(niveauxDeGris);
		JeuImages test = obtenirJeuChatChienTest(niveauxDeGris);

		if (train.entrees.length == 0 || test.entrees.length == 0) {
			System.err.println("Erreur : donnees chat/chien introuvables.");
			return;
		}

		System.out.printf(
			"Train chat/chien : %d images (%d chats, %d chiens).%n",
			train.entrees.length,
			train.nbChats,
			train.nbChiens);
		System.out.printf(
			"Test chat/chien : %d images (%d chats, %d chiens).%n",
			test.entrees.length,
			test.nbChats,
			test.nbChiens);

		modeleChatChien = entrainerModeleChatChien(train, activation, eta, epochsMax, mseLimite, true);
		modeleChatChien.precisionTest = evaluerScore(modeleChatChien.neurone, test.entrees, test.labels);

		System.out.printf("Score test apres entrainement : %.2f%%%n", modeleChatChien.precisionTest);
		System.out.println("IA chat/chien entrainee et gardee en memoire.");
	}

	private static void continuerEntrainementModeleCourant(Scanner clavier) {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages train = obtenirJeuChatChienTrain(modeleChatChien.niveauxDeGris);
		JeuImages test = obtenirJeuChatChienTest(modeleChatChien.niveauxDeGris);
		float eta = demanderFloatAvecDefaut(clavier, "Nouveau eta", 0.000001f, modeleChatChien.eta);
		int epochsMax = demanderEntierAvecDefaut(clavier, "Epochs supplementaires max", 1, modeleChatChien.epochsMax);
		float mseLimite = demanderFloatAvecDefaut(clavier, "MSE limite", 0.0f, modeleChatChien.mseLimite);

		Neurone.fixeCoefApprentissage(eta);
		Neurone.fixeNbEpochsMax(epochsMax);
		Neurone.fixeTraceApprentissage(true);

		System.out.println();
		System.out.println("=== REPRISE D'APPRENTISSAGE DU MODELE COURANT ===");
		modeleChatChien.neurone.apprentissage(train.entrees, train.labels, mseLimite);

		modeleChatChien.eta = eta;
		modeleChatChien.epochsMax = epochsMax;
		modeleChatChien.mseLimite = mseLimite;
		modeleChatChien.precisionTest = evaluerScore(modeleChatChien.neurone, test.entrees, test.labels);

		System.out.printf("Nouveau score test : %.2f%%%n", modeleChatChien.precisionTest);
	}

	private static void executerBenchmark(Scanner clavier) {
		System.out.println();
		System.out.println("=== BENCHMARK AUTOMATIQUE ===");
		System.out.println("Le benchmark teste plusieurs reglages et garde le meilleur");
		System.out.println("sur une partie validation extraite du train.");
		System.out.println();

		boolean niveauxDeGris = demanderRepresentationImages(clavier, true);
		String[] activations = demanderActivationsBenchmark(clavier);
		float[] etas = demanderListeFloatAvecDefaut(clavier, "Liste des eta separes par des virgules", "0.001,0.005,0.01");
		int repetitions = demanderEntierAvecDefaut(clavier, "Nombre de repetitions par configuration", 1, 1);
		int epochsMax = demanderEntierAvecDefaut(clavier, "Nombre maximal d'epochs", 1, EPOCHS_IMAGES_DEFAUT);
		float mseLimite = demanderFloatAvecDefaut(clavier, "MSE limite", 0.0f, MSE_LIMITE_IMAGES_DEFAUT);
		int pourcentValidation = demanderEntierAvecDefaut(clavier, "Pourcentage du train utilise en validation", 1, 20);

		JeuImages trainComplet = obtenirJeuChatChienTrain(niveauxDeGris);
		JeuImages testOfficiel = obtenirJeuChatChienTest(niveauxDeGris);
		int tailleValidation = Math.max(1, trainComplet.entrees.length * pourcentValidation / 100);
		int split = trainComplet.entrees.length - tailleValidation;

		if (split <= 0) {
			System.out.println("Pourcentage de validation trop grand pour le jeu d'entrainement.");
			return;
		}

		JeuImages trainBenchmark = sousJeu(trainComplet, 0, split);
		JeuImages validation = sousJeu(trainComplet, split, trainComplet.entrees.length);
		ModeleIA meilleur = null;

		System.out.println();
		System.out.println("Representation testee : " + nomRepresentation(niveauxDeGris));
		System.out.println("Activation | eta     | rep | validation | test officiel");
		System.out.println("--------------------------------------------------------");

		for (String activation : activations) {
			for (float eta : etas) {
				for (int rep = 1; rep <= repetitions; rep++) {
					ModeleIA modele = entrainerModeleChatChien(
						trainBenchmark,
						activation,
						eta,
						epochsMax,
						mseLimite,
						false);

					modele.precisionValidation = evaluerScore(modele.neurone, validation.entrees, validation.labels);
					modele.precisionTest = evaluerScore(modele.neurone, testOfficiel.entrees, testOfficiel.labels);

					System.out.printf(
						"%-10s | %.5f | %3d | %9.2f%% | %12.2f%%%n",
						modele.activation,
						modele.eta,
						rep,
						modele.precisionValidation,
						modele.precisionTest);

					if (meilleur == null || modele.precisionValidation > meilleur.precisionValidation) {
						meilleur = modele;
					}
				}
			}
		}

		Neurone.fixeTraceApprentissage(true);

		if (meilleur == null) {
			System.out.println("Aucun modele n'a ete entraine pendant le benchmark.");
			return;
		}

		modeleChatChien = meilleur;
		System.out.println();
		System.out.println("Meilleur modele garde en memoire :");
		afficherResumeModele(modeleChatChien);

		if (demanderOuiNon(clavier, "Sauvegarder ce meilleur modele ? (o/n) : ")) {
			sauvegarderModeleCourant();
		}
	}

	private static String[] demanderActivationsBenchmark(Scanner clavier) {
		System.out.println("Activations a tester :");
		System.out.println("1 - Heaviside seulement");
		System.out.println("2 - Sigmoide seulement");
		System.out.println("3 - ReLU seulement");
		System.out.println("4 - Les trois activations");

		while (true) {
			String choix = demanderTexteAvecDefaut(clavier, "Votre choix", "4");
			switch (choix) {
				case "1":
					return new String[] {"Heaviside"};
				case "2":
					return new String[] {"Sigmoide"};
				case "3":
					return new String[] {"ReLU"};
				case "4":
					return new String[] {"Heaviside", "Sigmoide", "ReLU"};
				default:
					System.out.println("Choix invalide. Entrez 1, 2, 3 ou 4.");
					break;
			}
		}
	}

	private static float[] demanderListeFloatAvecDefaut(Scanner clavier, String question, String valeurDefaut) {
		while (true) {
			String texte = demanderTexteAvecDefaut(clavier, question, valeurDefaut);
			String[] morceaux = texte.split("[,; ]+");
			List<Float> valeurs = new ArrayList<>();

			try {
				for (String morceau : morceaux) {
					if (!morceau.isEmpty()) {
						valeurs.add(Float.parseFloat(morceau.replace(",", ".")));
					}
				}
			} catch (NumberFormatException e) {
				System.out.println("Liste invalide. Exemple attendu : 0.001,0.005,0.01");
				continue;
			}

			if (!valeurs.isEmpty()) {
				float[] tableau = new float[valeurs.size()];
				for (int i = 0; i < valeurs.size(); i++) {
					tableau[i] = valeurs.get(i);
				}
				return tableau;
			}

			System.out.println("La liste ne doit pas etre vide.");
		}
	}

	private static ModeleIA entrainerModeleChatChien(
		JeuImages train,
		String activation,
		float eta,
		int epochsMax,
		float mseLimite,
		boolean afficherApprentissage) {

		int nbPixels = train.entrees[0].length;
		iNeurone neurone = creerNeurone(activation, nbPixels);

		Neurone.fixeCoefApprentissage(eta);
		Neurone.fixeNbEpochsMax(epochsMax);
		Neurone.fixeTraceApprentissage(afficherApprentissage);

		System.out.printf(
			"Apprentissage : activation=%s, representation=%s, eta=%.5f, epochsMax=%d, mseLimite=%.5f%n",
			activation,
			nomRepresentation(train.niveauxDeGris),
			eta,
			epochsMax,
			mseLimite);
		neurone.apprentissage(train.entrees, train.labels, mseLimite);

		return new ModeleIA(
			neurone,
			activation,
			eta,
			epochsMax,
			mseLimite,
			Double.NaN,
			Double.NaN,
			nbPixels,
			train.niveauxDeGris);
	}

	private static iNeurone creerNeurone(String activation, int nbEntrees) {
		if (activation.equalsIgnoreCase("Sigmoide")) {
			return new NeuroneSigmoide(nbEntrees);
		}
		if (activation.equalsIgnoreCase("ReLU")) {
			return new NeuroneReLU(nbEntrees);
		}
		return new NeuroneHeaviside(nbEntrees);
	}

	private static void testerIAChatChien(boolean afficherDetails, int limiteAffichage) {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages test = obtenirJeuChatChienTest(modeleChatChien.niveauxDeGris);

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
		afficherResumeModele(modeleChatChien);

		evaluerChatVsChien(modeleChatChien.neurone, test, afficherDetails, limiteAffichage);
		modeleChatChien.precisionTest = evaluerScore(modeleChatChien.neurone, test.entrees, test.labels);
	}

	private static void comparerTrainEtTest() {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages train = obtenirJeuChatChienTrain(modeleChatChien.niveauxDeGris);
		JeuImages test = obtenirJeuChatChienTest(modeleChatChien.niveauxDeGris);
		double scoreTrain = evaluerScore(modeleChatChien.neurone, train.entrees, train.labels);
		double scoreTest = evaluerScore(modeleChatChien.neurone, test.entrees, test.labels);

		System.out.println();
		System.out.println("=== COMPARAISON TRAIN / TEST ===");
		System.out.printf("Score sur train : %.2f%%%n", scoreTrain);
		System.out.printf("Score sur test  : %.2f%%%n", scoreTest);

		if (scoreTrain - scoreTest > 10.0) {
			System.out.println("Interpretation : ecart important, risque de surapprentissage.");
		} else {
			System.out.println("Interpretation : ecart raisonnable entre apprentissage et generalisation.");
		}
	}

	private static void testerRobustesseChatChien() {
		if (!iaChatChienPrete()) {
			return;
		}

		JeuImages test = obtenirJeuChatChienTest(modeleChatChien.niveauxDeGris);
		float[] amplitudes = {0.0f, 0.05f, 0.10f, 0.20f, 0.30f};

		System.out.println();
		System.out.println("=== ROBUSTESSE CHAT VS CHIEN AU BRUIT BLANC ===");
		System.out.println("Representation : " + nomRepresentation(modeleChatChien.niveauxDeGris));
		System.out.println("Amplitude | RSB (dB) | Fiabilite");

		for (float amplitude : amplitudes) {
			ResultatEvaluation resultat = evaluerAvecBruit(
				modeleChatChien.neurone,
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
		if (modeleChatChien != null) {
			return true;
		}

		System.out.println();
		System.out.println("IA chat/chien non entrainee.");
		System.out.println("Choisissez d'abord le menu 2 pour entrainer ou charger un modele.");
		return false;
	}

	private static void afficherResumeModele(ModeleIA modele) {
		System.out.printf("Activation : %s%n", modele.activation);
		System.out.printf("Representation : %s | entrees neurone : %d%n",
			nomRepresentation(modele.niveauxDeGris),
			modele.nbPixels);
		System.out.printf("eta : %.5f | epochsMax : %d | mseLimite : %.5f%n",
			modele.eta,
			modele.epochsMax,
			modele.mseLimite);

		if (!Double.isNaN(modele.precisionValidation)) {
			System.out.printf("Precision validation : %.2f%%%n", modele.precisionValidation);
		}
		if (!Double.isNaN(modele.precisionTest)) {
			System.out.printf("Precision test : %.2f%%%n", modele.precisionTest);
		}
	}

	private static DataPipeline obtenirPipelineImages(boolean niveauxDeGris) {
		if (niveauxDeGris && pipelineImagesGris != null) {
			return pipelineImagesGris;
		}
		if (!niveauxDeGris && pipelineImagesRGB != null) {
			return pipelineImagesRGB;
		}

		System.out.println();
		System.out.println("=== INITIALISATION DU PIPELINE ===");
		System.out.println("Representation : " + nomRepresentation(niveauxDeGris));

		// Dossier dataset : compatible lancement depuis racine projet ou depuis Support/
		String repTrain = premierRepertoireExistant("dataset_groupe_7/train", "../dataset_groupe_7/train");
		String repTest = premierRepertoireExistant("dataset_groupe_7/test", "../dataset_groupe_7/test");

		DataPipeline pipeline = new DataPipeline();
		pipeline.construire(repTrain, repTest, niveauxDeGris);

		if (niveauxDeGris) {
			pipelineImagesGris = pipeline;
		} else {
			pipelineImagesRGB = pipeline;
		}

		return pipeline;
	}

	private static JeuImages obtenirJeuChatChienTrain(boolean niveauxDeGris) {
		if (niveauxDeGris) {
			if (jeuChatChienTrainGris == null) {
				String repTrain = premierRepertoireExistant("dataset_groupe_7/train", "../dataset_groupe_7/train");
				jeuChatChienTrainGris = chargerJeuChatChien(repTrain, true, true);
			}

			return jeuChatChienTrainGris;
		}

		if (jeuChatChienTrainRGB == null) {
			String repTrain = premierRepertoireExistant("dataset_groupe_7/train", "../dataset_groupe_7/train");
			jeuChatChienTrainRGB = chargerJeuChatChien(repTrain, false, true);
		}

		return jeuChatChienTrainRGB;
	}

	private static JeuImages obtenirJeuChatChienTest(boolean niveauxDeGris) {
		if (niveauxDeGris) {
			if (jeuChatChienTestGris == null) {
				String repTest = premierRepertoireExistant("dataset_groupe_7/test", "../dataset_groupe_7/test");
				jeuChatChienTestGris = chargerJeuChatChien(repTest, true, false);
			}

			return jeuChatChienTestGris;
		}

		if (jeuChatChienTestRGB == null) {
			String repTest = premierRepertoireExistant("dataset_groupe_7/test", "../dataset_groupe_7/test");
			jeuChatChienTestRGB = chargerJeuChatChien(repTest, false, false);
		}

		return jeuChatChienTestRGB;
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
			return new JeuImages(new float[0][], new float[0], new String[0], 0, 0, niveauxDeGris);
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

		JeuImages jeu = convertirEnTableaux(entrees, labels, cheminsConserves, nbChats, nbChiens, niveauxDeGris);

		if (melanger) {
			melangerJeuImages(jeu);
		}

		System.out.printf(
			"Chargement chat/chien %s : %d images dans '%s'%n",
			nomRepresentation(niveauxDeGris),
			jeu.entrees.length,
			repertoire);
		return jeu;
	}

	private static JeuImages convertirEnTableaux(
		List<float[]> entrees,
		List<Float> labels,
		List<String> chemins,
		int nbChats,
		int nbChiens,
		boolean niveauxDeGris) {

		float[][] entreesTableau = new float[entrees.size()][];
		float[] labelsTableau = new float[labels.size()];
		String[] cheminsTableau = new String[chemins.size()];

		for (int i = 0; i < entrees.size(); i++) {
			entreesTableau[i] = entrees.get(i);
			labelsTableau[i] = labels.get(i);
			cheminsTableau[i] = chemins.get(i);
		}

		return new JeuImages(entreesTableau, labelsTableau, cheminsTableau, nbChats, nbChiens, niveauxDeGris);
	}

	private static JeuImages sousJeu(JeuImages source, int debut, int fin) {
		int taille = Math.max(0, fin - debut);
		float[][] entrees = new float[taille][];
		float[] labels = new float[taille];
		String[] chemins = new String[taille];
		int nbChats = 0;
		int nbChiens = 0;

		for (int i = 0; i < taille; i++) {
			int sourceIndex = debut + i;
			entrees[i] = source.entrees[sourceIndex];
			labels[i] = source.labels[sourceIndex];
			chemins[i] = source.chemins[sourceIndex];

			if (labels[i] >= 0.5f) {
				nbChiens++;
			} else {
				nbChats++;
			}
		}

		return new JeuImages(entrees, labels, chemins, nbChats, nbChiens, source.niveauxDeGris);
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

	private static void testerImagesSansBruit(boolean niveauxDeGris) {
		DataPipeline pipeline = obtenirPipelineImages(niveauxDeGris);
		iNeurone neurone = entrainerNeuroneImages(pipeline);

		if (neurone == null) {
			return;
		}

		System.out.println();
		System.out.println("=== EVALUATION SUR LE FLUX DE TEST ===");
		System.out.println("Representation : " + nomRepresentation(niveauxDeGris));

		double precision = evaluerScore(neurone, pipeline.getEntreesTest(), pipeline.getLabelsTest());
		System.out.printf("Precision finale sur le flux de Test : %.2f%%%n", precision);
		System.out.println("Interpretation : classification binaire chat vs non-chat.");
	}

	private static void testerRobustesseAvecBruit(boolean niveauxDeGris) {
		DataPipeline pipeline = obtenirPipelineImages(niveauxDeGris);
		iNeurone neurone = entrainerNeuroneImages(pipeline);

		if (neurone == null) {
			return;
		}

		float[] amplitudes = {0.0f, 0.05f, 0.10f, 0.20f, 0.30f};

		System.out.println();
		System.out.println("=== ROBUSTESSE AU BRUIT BLANC ===");
		System.out.println("Representation : " + nomRepresentation(niveauxDeGris));
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

	private static iNeurone entrainerNeuroneImages(DataPipeline pipeline) {
		float[][] entreesTrain = pipeline.getEntreesTrain();
		float[] labelsTrain = pipeline.getLabelsTrain();

		if (entreesTrain.length == 0) {
			System.err.println("Erreur : Donnees introuvables. Verifiez l'emplacement de 'dataset_groupe_7'.");
			return null;
		}

		int nbPixels = entreesTrain[0].length;
		System.out.printf("Nombre d'entrees du neurone : %d synapses.%n", nbPixels);

		Neurone.fixeCoefApprentissage(ETA_IMAGES_DEFAUT);
		Neurone.fixeNbEpochsMax(EPOCHS_IMAGES_DEFAUT);
		Neurone.fixeTraceApprentissage(true);

		// Couplage faible : on manipule le neurone via l'interface iNeurone.
		iNeurone neurone = new NeuroneHeaviside(nbPixels);

		System.out.println();
		System.out.println("=== DEBUT DE L'APPRENTISSAGE ===");
		neurone.apprentissage(entreesTrain, labelsTrain, MSE_LIMITE_IMAGES_DEFAUT);

		return neurone;
	}

	private static double evaluerScore(iNeurone neurone, float[][] entrees, float[] labels) {
		int succes = 0;

		for (int i = 0; i < entrees.length; i++) {
			neurone.metAJour(entrees[i]);
			float prediction = predictionBinaire(neurone);

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
			float prediction = predictionBinaire(neurone);
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
			float prediction = predictionBinaire(neurone);

			if (prediction == labels[i]) {
				succes++;
			}
		}

		double precision = (double) succes / entrees.length * 100.0;
		double rsb = calculerRSB(puissanceSignal / nbPixels, puissanceBruit / nbPixels);

		return new ResultatEvaluation(precision, rsb);
	}

	private static float predictionBinaire(iNeurone neurone) {
		return neurone.sortie() >= 0.5f ? 1.0f : 0.0f;
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

	private static void sauvegarderModeleCourant() {
		if (!iaChatChienPrete()) {
			return;
		}

		try {
			Files.createDirectories(Paths.get(DOSSIER_MODELES));
			modeleChatChien.neurone.sauvegarde(FICHIER_POIDS_MODELE);

			Properties meta = new Properties();
			meta.setProperty("activation", modeleChatChien.activation);
			meta.setProperty("eta", String.valueOf(modeleChatChien.eta));
			meta.setProperty("epochsMax", String.valueOf(modeleChatChien.epochsMax));
			meta.setProperty("mseLimite", String.valueOf(modeleChatChien.mseLimite));
			meta.setProperty("precisionValidation", String.valueOf(modeleChatChien.precisionValidation));
			meta.setProperty("precisionTest", String.valueOf(modeleChatChien.precisionTest));
			meta.setProperty("nbPixels", String.valueOf(modeleChatChien.nbPixels));
			meta.setProperty("nbEntrees", String.valueOf(modeleChatChien.nbPixels));
			meta.setProperty("niveauxDeGris", String.valueOf(modeleChatChien.niveauxDeGris));
			meta.setProperty("representation", nomRepresentation(modeleChatChien.niveauxDeGris));

			try (FileWriter writer = new FileWriter(FICHIER_META_MODELE)) {
				meta.store(writer, "Modele chat vs chien sauvegarde");
			}

			System.out.println("Modele sauvegarde dans le dossier '" + DOSSIER_MODELES + "'.");
		} catch (IOException e) {
			System.out.println("Impossible de sauvegarder le modele.");
			e.printStackTrace();
		}
	}

	private static boolean chargerModeleSauvegarde(boolean afficherMessage) {
		Path poids = Paths.get(FICHIER_POIDS_MODELE);
		Path meta = Paths.get(FICHIER_META_MODELE);

		if (!Files.isRegularFile(poids) || !Files.isRegularFile(meta)) {
			if (afficherMessage) {
				System.out.println("Aucun modele sauvegarde trouve.");
			}
			return false;
		}

		try (FileReader reader = new FileReader(FICHIER_META_MODELE)) {
			Properties proprietes = new Properties();
			proprietes.load(reader);

			String activation = proprietes.getProperty("activation", "Heaviside");
			float eta = Float.parseFloat(proprietes.getProperty("eta", String.valueOf(ETA_IMAGES_DEFAUT)));
			int epochsMax = Integer.parseInt(proprietes.getProperty("epochsMax", String.valueOf(EPOCHS_IMAGES_DEFAUT)));
			float mseLimite = Float.parseFloat(proprietes.getProperty("mseLimite", String.valueOf(MSE_LIMITE_IMAGES_DEFAUT)));
			double precisionValidation = Double.parseDouble(proprietes.getProperty("precisionValidation", "NaN"));
			double precisionTest = Double.parseDouble(proprietes.getProperty("precisionTest", "NaN"));
			int nbPixels = Integer.parseInt(proprietes.getProperty("nbEntrees", proprietes.getProperty("nbPixels", "4096")));
			boolean niveauxDeGris = Boolean.parseBoolean(proprietes.getProperty("niveauxDeGris", "true"));

			iNeurone neurone = creerNeurone(activation, nbPixels);
			neurone.chargement(FICHIER_POIDS_MODELE);
			modeleChatChien = new ModeleIA(
				neurone,
				activation,
				eta,
				epochsMax,
				mseLimite,
				precisionValidation,
				precisionTest,
				nbPixels,
				niveauxDeGris);

			if (afficherMessage) {
				System.out.println("Modele sauvegarde charge en memoire.");
				afficherResumeModele(modeleChatChien);
			}

			return true;
		} catch (Exception e) {
			System.out.println("Impossible de charger le modele sauvegarde.");
			e.printStackTrace();
			return false;
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

	private static String nomClasseChatChien(float label) {
		return label >= 0.5f ? "chien" : "chat";
	}

	private static String nomFichier(String chemin) {
		return Paths.get(chemin).getFileName().toString();
	}
}
