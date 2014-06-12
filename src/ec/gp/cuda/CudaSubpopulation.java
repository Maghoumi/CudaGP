package ec.gp.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ec.EvolutionState;
import ec.Individual;
import ec.Subpopulation;
import ec.gp.GPIndividual;
import ec.util.Parameter;

/**
 * An exact copy of ec.Sobpopulation. However, after initializing the
 * population, the initial population is added to the list of the current
 * thread's unevaluated individuals. This way, we can easily convert the
 * list of unevaluated individuals to byte expressions without checking
 * the evaluated flag of each individual. Smart, eh? :-)
 * 
 * @author Mehran Maghoumi
 * 
 */
public class CudaSubpopulation extends Subpopulation {

	private static final long serialVersionUID = -6398978647662251983L;
	
	/**
	 * A list of individuals that require evaluation. To prevent locking and
	 * synchronization each thread has its own list
	 */
	public List<List<GPIndividual>> needEval;

	@Override
	public void setup(EvolutionState state, Parameter base) {
		super.setup(state, base);

		// Initialize the unevaluated lists of this subpopulation
		needEval = new ArrayList<List<GPIndividual>>(state.breedthreads);

		for (int i = 0; i < state.breedthreads; i++) {
			needEval.add(new ArrayList<GPIndividual>());
		}

	}

	public void populate(EvolutionState state, int thread) {

		List<GPIndividual> myUnevals = this.needEval.get(thread);
		int len = individuals.length; // original length of individual array
		int start = 0; // where to start filling new individuals in -- may get
						// modified if we read some individuals in

		// should we load individuals from a file? -- duplicates are permitted
		if (loadInds) {
			InputStream stream = state.parameters.getResource(file, null);
			if (stream == null)
				state.output.fatal("Could not load subpopulation from file", file);

			try {
				readSubpopulation(state, new LineNumberReader(new InputStreamReader(stream)));
			} catch (IOException e) {
				state.output.fatal("An IOException occurred when trying to read from the file " + state.parameters.getString(file, null) + ".  The IOException was: \n" + e, file, null);
			}

			if (len < individuals.length) {
				state.output.message("Old subpopulation was of size " + len + ", expanding to size " + individuals.length);
				return;
			}

			if (len > individuals.length) // the population was shrunk, there's
											// more space yet
			{
				// What do we do with the remainder?
				if (extraBehavior == TRUNCATE) {
					state.output.message("Old subpopulation was of size " + len + ", truncating to size " + individuals.length);
					return; // we're done
				} else if (extraBehavior == WRAP) {
					state.output.message("Only " + individuals.length + " individuals were read in.  Subpopulation will stay size " + len + ", and the rest will be filled with copies of the read-in individuals.");

					Individual[] oldInds = individuals;
					individuals = new Individual[len];
					System.arraycopy(oldInds, 0, individuals, 0, oldInds.length);
					start = oldInds.length;

					int count = 0;
					for (int i = start; i < individuals.length; i++) {
						individuals[i] = (Individual) (individuals[count].clone());
						if (++count >= start)
							count = 0;
					}
					return;
				} else // if (extraBehavior == FILL)
				{
					state.output.message("Only " + individuals.length + " individuals were read in.  Subpopulation will stay size " + len + ", and the rest will be filled using randomly generated individuals.");

					Individual[] oldInds = individuals;
					individuals = new Individual[len];
					System.arraycopy(oldInds, 0, individuals, 0, oldInds.length);
					start = oldInds.length;
					// now go on to fill the rest below...
				}
			}
		}
		
		// populating the remainder with random individuals
		HashMap h = null;
		if (numDuplicateRetries >= 1)
			h = new HashMap((individuals.length - start) / 2); // seems
																// reasonable

		for (int x = start; x < individuals.length; x++) {
			for (int tries = 0; tries <= /* Yes, I see that */numDuplicateRetries; tries++) {
				individuals[x] = species.newIndividual(state, thread);

				if (numDuplicateRetries >= 1) {
					// check for duplicates
					Object o = h.get(individuals[x]);
					if (o == null) // found nothing, we're safe
					// hash it and go
					{
						// Add this individual to the current thread's
						// unevaluated individuals
						myUnevals.add((GPIndividual) individuals[x]);
						h.put(individuals[x], individuals[x]);
						break;
					}
				}
			} // oh well, we tried to cut down the duplicates
		}
	}
}
