package ec.gp.cuda;

import jcuda.Pointer;
import jcuda.driver.CUmodule;
import ec.gp.GPData;

/**
 * An extension of GPData which stores the data that should be passed
 * to individuals for evaluation. Subclasses of this class can use this class
 * to store GP data and also prepare CUDA-side data (such as input arrays for the
 * kernel function).
 * This class was declared abstract because I needed a means of forcing the user
 * to provide a means of obtaining CUDA pointers to the allocated data as well as
 * to provide the operations that should be carried out before and after a kernel
 * invocation. This way we can call the kernel function in CUDAInterop.
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class CudaData extends GPData {

	private static final long serialVersionUID = -6076018909354741619L;
	
	/**
	 * @return An array of JCuda Pointer[]. Each element of the array contains
	 * 			a pointer to a kernel argument.
	 * <b>IMPORTANT NOTE:</b>
	 * 		The order of the pointers in the returned array matters
	 * 		and they should be in the same order defined in ECJ's
	 * 		parameter file and the same order that the kernel function
	 * 		expects them. You should only provide a pointer to you own
	 * 		defined kernel arguments. Other arguments (such as the fitnesses[]
	 * 		array will be taken care of automatically by CudaEvaluator.
	 * 
	 */
	public abstract Pointer[] getArgumentPointers();
	
	/**
	 * Defines the operations that should be done before a CUDA kernel is invoked.
	 * This usually includes allocating device memories or preparing data structures. 
	 * @param module	The CUDA module to use in case any texture allocation operations
	 * 					are required
	 */
	public abstract void preInvocationTasks(CUmodule module);
	
	/**
	 * Defines the operations that should be done after a CUDA kernel is invoked.
	 * This usually includes freeing device memory and refreshing the arrays that were
	 * passed to the kernel. 
	 * @param module	The CUDA module to use in case any texture allocation operations
	 * 					are required
	 */
	public abstract void postInvocationTasks(CUmodule module);
}
