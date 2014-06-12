/**
 * This is a generic evaluator template. This file should be manually edited
 * and adapted to the problem at hand
 *
 * @author Mehran Maghoumi
 *
 */

/** =====================================Stack related definitions==================================== */
/** The size of the interpreter stack */
#define STACK_SIZE /*@@stack-size@@*/
#define push(A) do { sp++;stack[sp]=A; if(sp >= STACK_SIZE) printf("Stack overflow");} while(false)
#define pop(A) do{ A=stack[sp];sp--; if(sp < -1) printf("Stack underflow");} while(false)
/** ================================================================================================== */

/** The number of training instances that the individual is to be evaluated for */
#define PROBLEM_SIZE /*@@problem-size@@*/
#define BLOCK_SIZE /*@@block-size@@*/	// Used for the shared memory definitions


/************************************************************************************************************
 ************************************************************************************************************/

//TODO DOC: sadly there is only support for 1 pitch value for all input instances (which should be more than enough)
extern "C"
__global__ void evaluate(/*@@kernel-args@@*/ int inputPitch,
						/*@@kernel-out@@*/ int outputPitch,
						const char* __restrict__ individuals, const int indCounts, const int maxLength)
{
	int blockIndex = blockIdx.x;
	int threadIndex = threadIdx.x;

	if (blockIndex >= indCounts)
		return;

	// Obtain pointer to the beginning of the memory space of the individual that
	// this block will evaluate
	const char* __restrict__ expression = &(individuals[blockIndex * maxLength]);
	/*@@kernel-out-type@@*/ blockOutput = &(/*@@kernel-out-name@@*/[blockIndex * outputPitch]);

	// the first thread should reset these values
//	if (threadIndex == 0) {
//		fitnesses[blockIndex] = 0;
//	}

	/*@@kernel-out-type-nopointer@@*/ stack[STACK_SIZE];	// The stack is defined as the same type as the kernel output
	int sp;

	// Determine how many fitness cases this thread should process
	int portion = (PROBLEM_SIZE - 1)/ blockDim.x  + 1;

	for (int i = 0 ; i < portion; i++) {

		// Thread to data index mapping with respect to the loop variable
		int tid = portion * threadIndex + i;

		if (tid >= PROBLEM_SIZE)
			break;

		// Reset the stack pointer
		sp = - 1;

		int k = 0;	// Maintains the current index in the expression
		while(expression[k] != 0)
		{
			switch(expression[k])
			{
				/*@@interpreter@@*/
			}

			k++;
		}

		// Pop the top of the stack
		/*@@kernel-out-type-nopointer@@*/ stackTop;
		pop(stackTop);

		if(sp!=-1)
			printf("Stack pointer is not -1 but is %d", sp);

		// Assign the top of the stack to the output
		blockOutput[tid] = stackTop;

	}
}
