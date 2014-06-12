/**
 * This is a generic evaluator template. This file should be manually edited
 * and adapted to the problem at hand
 *
 * @author Mehran Maghoumi
 *
 */

/** =====================================Stack related definitions==================================== */
/** The size of the interpreter stack */
#define STACK_SIZE 128
#define push(A) do { sp++;stack[sp]=A; if(sp >= STACK_SIZE) printf("Stack overflow");} while(false)
#define pop(A) do{ A=stack[sp];sp--; if(sp < -1) printf("Stack underflow");} while(false)
/** ================================================================================================== */

/** The number of training instances that the individual is to be evaluated for */
#define PROBLEM_SIZE 9
#define BLOCK_SIZE 512	// Used for the shared memory definitions


/************************************************************************************************************
 ************************************************************************************************************/

/**
 * Does a vector reduction with the sum operation. This is usually required for most GP problems.
 * Feel free to use this function or other functions of your own.
 */
//FIXME might not work for odd number of inputs
__device__ inline void reduce(double *input) {
	__syncthreads();

	for (int stride = blockDim.x >> 1 ; stride > 0 ; stride >>= 1) {
		if (threadIdx.x < stride) {
			input[threadIdx.x]+= input[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (blockDim.x % 2 != 0) {	// For odd number of inputs
		if (threadIdx.x == 0) {
			input[0] += input[blockDim.x - 1];	// Add the last element by the first thread
		}
	}

}

extern "C"
__global__ void evaluate(double* x, double* y, double* expected, 
		const char* __restrict__ individuals, const int indCounts, const int maxLength,
		double *fitnesses)
{
	int blockIndex = blockIdx.x;
	int threadIndex = threadIdx.x;

	if (blockIndex >= indCounts)
		return;

	const char* __restrict__ expression = &(individuals[blockIndex * maxLength]);

	// the first thread should reset these values
	if (threadIndex == 0) {
		fitnesses[blockIndex] = 0;
	}

	double stack[STACK_SIZE];
	int sp;

	// Determine how many fitness cases this thread should process
	int portion = (PROBLEM_SIZE - 1)/ blockDim.x  + 1;


	/*@@pre-eval-declare-provider@@*/
	/**/
	/**/__shared__ double sum[BLOCK_SIZE];
	/**/sum[threadIndex] = 0;

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
				case 1: {push(x[tid]);
}break;
case 2: {push(y[tid]);
}break;
case 3: {double second;pop(second);
double first;pop(first);
double final = first + second;
push(final);
}break;
case 4: {double second;pop(second);
double first;pop(first);
double final = first - second;
push(final);
}break;
case 5: {double second;pop(second);
double first;pop(first);
double final = first * second;
push(final);
}break;
default:printf("Unrecognized OPCODE in the expression tree!");break;
			}

			k++;
		}

		double eval_result;
		pop(eval_result);

		if(sp!=-1)
			printf("Stack pointer not -1 but is %d", sp);

		/*@@fitness-for-current-test-case@@*/
		/**/double expectedResult = expected[tid];
		/**/sum[threadIndex] += abs(expectedResult - eval_result);
	}

	/*@@calculate-fitness@@*/
	/**///Should reduce the shared memory

	/**/double PROBABLY_ZERO = 1.11E-15;
	/**/if (sum[threadIndex] < PROBABLY_ZERO)
		/**/sum[threadIndex] = (double)0.0;

	reduce(sum);

	// calculate the total fitness and assign it
	if (threadIndex == 0) {
		/*@@assign-fitness@@*/
		fitnesses[blockIndex] = sum[0];
	}
}