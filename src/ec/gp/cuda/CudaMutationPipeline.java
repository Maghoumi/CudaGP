package ec.gp.cuda;

import java.util.List;

import ec.BreedingPipeline;
import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPBreedingPipeline;
import ec.gp.GPIndividual;
import ec.gp.GPInitializer;
import ec.gp.GPNode;
import ec.gp.GPNodeBuilder;
import ec.gp.GPNodeSelector;
import ec.gp.GPTree;
import ec.gp.koza.GPKozaDefaults;
import ec.util.Parameter;

/**
 * A mutation pipeline that mimics the standard mutation pipeline in ECJ
 * However, after each modification, the modified individual is also stored in a
 * list of individuals that are not evaluated. This will make threading easier and
 * will speedup the system a little bit.
 * 
 * Since the original MutationPipeline could not be extended via intehritence
 * (some fields were defined as private) I had to copy the code over here...
 * 
 * @author Mehran Maghoumi
 * 
 */
public class CudaMutationPipeline extends GPBreedingPipeline {
	private static final long serialVersionUID = -4822119204988279354L;
	
	public static final String P_NUM_TRIES = "tries";
	public static final String P_MAXDEPTH = "maxdepth";
	public static final String P_MUTATION = "mutate";
	public static final String P_BUILDER = "build";
	public static final String P_EQUALSIZE = "equal";
	public static final int INDS_PRODUCED = 1;
	public static final int NUM_SOURCES = 1;

	/** How the pipeline chooses a subtree to mutate */
	public GPNodeSelector nodeselect;

	/** How the pipeline builds a new subtree */
	public GPNodeBuilder builder;

	/**
	 * The number of times the pipeline tries to build a valid mutated tree
	 * before it gives up and just passes on the original
	 */
	int numTries;

	/** The maximum depth of a mutated tree */
	int maxDepth;

	/** Do we try to replace the subtree with another of the same size? */
	boolean equalSize;

	/** Is our tree fixed? If not, this is -1 */
	int tree;

	public Parameter defaultBase() {
		return GPKozaDefaults.base().push(P_MUTATION);
	}

	public int numSources() {
		return NUM_SOURCES;
	}

	public Object clone() {
		CudaMutationPipeline c = (CudaMutationPipeline) (super.clone());

		// deep-cloned stuff
		c.nodeselect = (GPNodeSelector) (nodeselect.clone());

		return c;
	}

	public void setup(final EvolutionState state, final Parameter base) {
		super.setup(state, base);

		Parameter def = defaultBase();
		Parameter p = base.push(P_NODESELECTOR).push("" + 0);
		Parameter d = def.push(P_NODESELECTOR).push("" + 0);

		nodeselect = (GPNodeSelector) (state.parameters
				.getInstanceForParameter(p, d, GPNodeSelector.class));
		nodeselect.setup(state, p);

		p = base.push(P_BUILDER).push("" + 0);
		d = def.push(P_BUILDER).push("" + 0);

		builder = (GPNodeBuilder) (state.parameters.getInstanceForParameter(p,
				d, GPNodeBuilder.class));
		builder.setup(state, p);

		numTries = state.parameters.getInt(base.push(P_NUM_TRIES),
				def.push(P_NUM_TRIES), 1);
		if (numTries == 0)
			state.output
					.fatal("Mutation Pipeline has an invalid number of tries (it must be >= 1).",
							base.push(P_NUM_TRIES), def.push(P_NUM_TRIES));

		maxDepth = state.parameters.getInt(base.push(P_MAXDEPTH),
				def.push(P_MAXDEPTH), 1);
		if (maxDepth == 0)
			state.output.fatal("The Mutation Pipeline " + base
					+ "has an invalid maximum depth (it must be >= 1).",
					base.push(P_MAXDEPTH), def.push(P_MAXDEPTH));

		equalSize = state.parameters.getBoolean(base.push(P_EQUALSIZE),
				def.push(P_EQUALSIZE), false);

		tree = TREE_UNFIXED;
		if (state.parameters.exists(base.push(P_TREE).push("" + 0),
				def.push(P_TREE).push("" + 0))) {
			tree = state.parameters.getInt(base.push(P_TREE).push("" + 0), def
					.push(P_TREE).push("" + 0), 0);
			if (tree == -1)
				state.output
						.fatal("Tree fixed value, if defined, must be >= 0");
		}
	}

	/** Returns true if inner1 can feasibly be swapped into inner2's position */

	public boolean verifyPoints(GPNode inner1, GPNode inner2) {
		// We know they're swap-compatible since we generated inner1
		// to be exactly that. So don't bother.

		// next check to see if inner1 can fit in inner2's spot
		if (inner1.depth() + inner2.atDepth() > maxDepth)
			return false;

		// checks done!
		return true;
	}

	public int produce(final int min, final int max, final int start,
			final int subpopulation, final Individual[] inds,
			final EvolutionState state, final int thread) {
		// grab individuals from our source and stick 'em right into inds.
		// we'll modify them from there
		int n = sources[0].produce(min, max, start, subpopulation, inds, state,
				thread);

		// should we bother?
		if (!state.random[thread].nextBoolean(likelihood))
			return reproduce(n, start, subpopulation, inds, state, thread,
					false); // DON'T produce children from source -- we already
							// did

		GPInitializer initializer = ((GPInitializer) state.initializer);

		// now let's mutate 'em
		for (int q = start; q < n + start; q++) {
			GPIndividual i = (GPIndividual) inds[q];

			if (tree != TREE_UNFIXED && (tree < 0 || tree >= i.trees.length))
				// uh oh
				state.output
						.fatal("GP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees.  Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual");

			int t;
			// pick random tree
			if (tree == TREE_UNFIXED)
				if (i.trees.length > 1)
					t = state.random[thread].nextInt(i.trees.length);
				else
					t = 0;
			else
				t = tree;

			// validity result...
			boolean res = false;

			// prepare the nodeselector
			nodeselect.reset();

			// pick a node

			GPNode p1 = null; // the node we pick
			GPNode p2 = null;

			for (int x = 0; x < numTries; x++) {
				// pick a node in individual 1
				p1 = nodeselect.pickNode(state, subpopulation, thread, i,
						i.trees[t]);

				// generate a tree swap-compatible with p1's position

				int size = GPNodeBuilder.NOSIZEGIVEN;
				if (equalSize)
					size = p1.numNodes(GPNode.NODESEARCH_ALL);

				p2 = builder.newRootedTree(state, p1.parentType(initializer),
						thread, p1.parent,
						i.trees[t].constraints(initializer).functionset,
						p1.argposition, size);

				// check for depth and swap-compatibility limits
				res = verifyPoints(p2, p1); // p2 can fit in p1's spot -- the
											// order is important!

				// did we get something that had both nodes verified?
				if (res)
					break;
			}

			GPIndividual j;

			CudaSubpopulation subPop = (CudaSubpopulation) state.population.subpops[subpopulation];
            List<GPIndividual> myList = subPop.needEval.get(thread);

			if (sources[0] instanceof BreedingPipeline)
			// it's already a copy, so just smash the tree in
			{
				j = i;
				if (res) // we're in business
				{
					p2.parent = p1.parent;
					p2.argposition = p1.argposition;
					if (p2.parent instanceof GPNode)
						((GPNode) (p2.parent)).children[p2.argposition] = p2;
					else
						((GPTree) (p2.parent)).child = p2;
					j.evaluated = false; // we've modified it
					myList.add(j);
				}
			} else // need to clone the individual
			{
				j = (GPIndividual) (i.lightClone());

				// Fill in various tree information that didn't get filled in
				// there
				j.trees = new GPTree[i.trees.length];

				// at this point, p1 or p2, or both, may be null.
				// If not, swap one in. Else just copy the parent.
				for (int x = 0; x < j.trees.length; x++) {
					if (x == t && res) // we've got a tree with a kicking cross
										// position!
					{
						j.trees[x] = (GPTree) (i.trees[x].lightClone());
						j.trees[x].owner = j;
						j.trees[x].child = i.trees[x].child
								.cloneReplacingNoSubclone(p2, p1);
						j.trees[x].child.parent = j.trees[x];
						j.trees[x].child.argposition = 0;
						j.evaluated = false;
						myList.add(j);
					} // it's changed
					else {
						j.trees[x] = (GPTree) (i.trees[x].lightClone());
						j.trees[x].owner = j;
						j.trees[x].child = (GPNode) (i.trees[x].child.clone());
						j.trees[x].child.parent = j.trees[x];
						j.trees[x].child.argposition = 0;
					}
				}
			}

			// add the new individual, replacing its previous source
			inds[q] = j;
		}
		return n;
	}
}
