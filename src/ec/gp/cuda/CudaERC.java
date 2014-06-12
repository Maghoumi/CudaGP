package ec.gp.cuda;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ec.EvolutionState;
import ec.gp.ERC;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import ec.util.DecodeReturn;
import ec.util.Parameter;

/**
 * Basically the exact same code from ec.gp.ERC class Except that this class
 * inherits from CudaNode rather than GPNode
 * 
 * @author Mehran Maghoumi
 * 
 */
public abstract class CudaERC extends CudaNode {

	private static final long serialVersionUID = 1195782977469434888L;

	/**
	 * Returns the lowercase "name" of this ERC function class, some simple,
	 * short name which distinguishes this class from other ERC function classes
	 * you're using. If you have more than one ERC function, you need to
	 * distinguish them here. By default the value is "ERC", which works fine
	 * for a single ERC function in the function set. Whatever the name is, it
	 * should generally only have letters, numbers, or hyphens or underscores in
	 * it. No whitespace or other characters.
	 */
	public String name() {
		return "ERC";
	}

	/**
	 * Usually ERCs don't have children, and this default implementation makes
	 * certain of it. But if you want to override this, you're welcome to.
	 */
	public void checkConstraints(final EvolutionState state,
			final int tree,
			final GPIndividual typicalIndividual,
			final Parameter individualBase)
	{
		super.checkConstraints(state, tree, typicalIndividual, individualBase);
		// make sure we don't have any children.  This is the typical situation for an ERC.
		if (children.length != 0)
			state.output.error("Incorrect number of children for the node " + toStringForError() + " (should be 0)");
	}

	/**
	 * Remember to override this to randomize your ERC after it has been cloned.
	 * The prototype will not ever receive this method call.
	 */
	public abstract void resetNode(final EvolutionState state, int thread);

	/** Implement this to do ERC-to-ERC comparisons. */
	public abstract boolean nodeEquals(final GPNode node);

	/**
	 * Implement this to hash ERCs, along with other nodes, in such a way that
	 * two "equal" ERCs will usually hash to the same value. The default value,
	 * which may not be very good, is a combination of the class hash code and
	 * the hash code of the string returned by encode(). You might make a better
	 * hash value.
	 */
	public int nodeHashCode() {
		return super.nodeHashCode() ^ encode().hashCode();
	}

	/**
	 * You might want to override this to return a special human-readable
	 * version of the erc value; otherwise this defaults to toString(); This
	 * should be something that resembles a LISP atom. If a simple number or
	 * other object won't suffice, you might use something that begins with
	 * name() + [ + ... + ]
	 */
	public String toStringForHumans()
	{
		return toString();
	}

	/**
	 * This defaults to simply name() + "[" + encode() + "]". You probably
	 * shouldn't deviate from this.
	 */
	public String toString()
	{
		return name() + "[" + encode() + "]";
	}

	/** Encodes data from the ERC, using ec.util.Code. */
	public abstract String encode();

	/**
	 * Decodes data into the ERC from dret. Return true if you sucessfully
	 * decoded, false if you didn't. Don't increment dret.pos's value beyond
	 * exactly what was needed to decode your ERC. If you fail to decode, you
	 * should make sure that the position and data in the dret are exactly as
	 * they were originally.
	 */
	public boolean decode(final DecodeReturn dret)
	{
		return false;
	}

	/**
	 * Mutates the node's "value". This is called by mutating operators which
	 * specifically <i>mutate</i> the "value" of ERCs, as opposed to replacing
	 * them with whole new ERCs. The default form of this function simply calls
	 * resetNode(state,thread), but you might want to modify this to do a
	 * specialized form of mutation, applying gaussian noise for example.
	 */

	public void mutateERC(final EvolutionState state, final int thread)
	{
		resetNode(state, thread);
	}

	/**
	 * To successfully write to a DataOutput, you must override this to write
	 * your specific ERC data out. The default implementation issues a fatal
	 * error.
	 */
	public void writeNode(final EvolutionState state, final DataOutput dataOutput) throws IOException
	{
		state.output.fatal("writeNode(EvolutionState,DataInput) not implemented in " + getClass().getName());
	}

	/**
	 * To successfully read from a DataOutput, you must override this to read
	 * your specific ERC data in. The default implementation issues a fatal
	 * error.
	 */
	public void readNode(final EvolutionState state, final DataInput dataInput) throws IOException
	{
		state.output.fatal("readNode(EvolutionState,DataInput) not implemented in " + getClass().getName());
	}

	public GPNode readNode(final DecodeReturn dret)
	{
		int len = dret.data.length();
		int originalPos = dret.pos;

		// get my name
		String str2 = name() + "[";
		int len2 = str2.length();

		if (dret.pos + len2 >= len) // uh oh, not enough space
			return null;

		// check it out
		for (int x = 0; x < len2; x++)
			if (dret.data.charAt(dret.pos + x) != str2.charAt(x))
				return null;

		// looks good!  try to load this sucker.
		dret.pos += len2;
		ERC node = (ERC) lightClone();
		if (!node.decode(dret))
		{
			dret.pos = originalPos;
			return null;
		} // couldn't decode it

		// the next item should be a "]"

		if (dret.pos >= len)
		{
			dret.pos = originalPos;
			return null;
		}
		if (dret.data.charAt(dret.pos) != ']')
		{
			dret.pos = originalPos;
			return null;
		}

		// Check to make sure that the ERC's all there is
		if (dret.data.length() > dret.pos + 1)
		{
			char c = dret.data.charAt(dret.pos + 1);
			if (!Character.isWhitespace(c) &&
					c != ')' && c != '(') // uh oh
			{
				dret.pos = originalPos;
				return null;
			}
		}

		dret.pos++;

		return node;
	}

}
