package Support.neurone;

public class NeuroneSigmoide extends Neurone
{
	// Fonction d'activation sigmoïde bornée entre 0 et 1
	protected float activation(final float valeur) {return 1.f / (1.f + (float)Math.exp(-valeur));}

	// Constructeur
	public NeuroneSigmoide(final int nbEntrees) {super(nbEntrees);}
}
