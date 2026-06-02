package Support.neurone;

public class NeuroneHeaviside extends Neurone
{
    @Override
    protected float activation(final float valeur) {
        return valeur >= 0.0f ? 1.0f : 0.0f;
    }

    public NeuroneHeaviside(final int nbEntrees) {
        super(nbEntrees);
    }
}
