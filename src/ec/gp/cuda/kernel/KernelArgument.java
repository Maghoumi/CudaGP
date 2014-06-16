package ec.gp.cuda.kernel;

import ec.gp.cuda.CudaType;
import ec.gp.cuda.KernelOutputData;

/**
 * Defines the properties of an argument of the CUDA evaluation kernel.
 * In case this represents a kernel output argument, the class of the instance
 * that is specified in ECJ's parameter file must be castable to KernelOutputData 
 * The class name of the instances of this class must be castable to KernelOutputData
 * 
 * @author Mehran Maghoumi
 *
 */
public class KernelArgument {
	/** The name of this output variable */
	protected String name;
	
	/** The type of this output variable */
	protected CudaType type;
	
	/** The class name that should be used for interoperability with this output */
	protected String className;
	
	/** Flag indicating whether this argument is a pointer */
	protected boolean pointer;
	
	public KernelArgument(String name, String type, String className) {
		this.name = name;
		this.type = new CudaType(type);
		this.pointer = type.contains("*");
		this.className = className;
	}
	
	public KernelArgument(String name, String type) {
		this.name = name;
		this.type = new CudaType(type);
		this.pointer = type.contains("*");
		this.className = null;
	}
	
	/**
	 * @return	The name of this output argument
	 */
	public String getName() {
		return this.name;
	}
	
	/**
	 * @return	The type of this output argument
	 */
	public CudaType getType() {
		return this.type;
	}
	
	/**
	 * @return	True of this argument is a pointer, False otherwise
	 */
	public boolean isPointer() {
		return this.pointer;
	}
	
	/**
	 * @return	A string representing the correct C syntax that is used
	 * 			in the kernel signature
	 */
	public String getKernelSignature() {
		return String.format("%s %s,", type.getName(), name);
	}
	
	/**
	 * @param problemSize
	 * 			The size of the problem to use for initializing the new instance
	 * 			of the KernelOutputData subclass (the interop class) 				
	 * @return	An instance of the interoperability class that works with the kernel's output
	 * 			argument of this type.
	 */
	public KernelOutputData getInstanceForOutput(int problemSize) {
		try {
			Class<?> result = Class.forName(className, true, Thread.currentThread().getContextClassLoader());
			KernelOutputData newInstance = (KernelOutputData) result.newInstance();
			newInstance.init(problemSize);
			return newInstance;
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		}
		
		return null;
	}
}
