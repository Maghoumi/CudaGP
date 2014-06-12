package ec.gp.cuda;

import java.util.Stack;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import gnu.trove.list.TByteList;
import gnu.trove.list.array.TByteArrayList;

/**
 * An extension to ECJ's GPNode class. The CudaNode class, represents all
 * GPNodes in the system that could be used with JCuda. The nodes of 
 * this type can generate OPCODEs that can be interpretted by a CUDA postifx
 * evaluator. In addition, the nodes of type CudaNode must implement in CUDA-C 
 * the micro-operations that are required to evaluate them. An example would be
 * a simple ADD node. To implement add in CUDA, the following operations are
 * necessary:
 * 	1) pop the first operand from the postifx evaluation stack
 * 	2) pop the second operand from the postifx evaluation stack
 * 	3) add the two popped values together
 * 	4) push the result of the addition onto the postix evaluation stack
 * 
 * As of this moment, all micro-operations should be written in pure CUDA-C,
 * honoring the programmer's predefined contracts. IE. for this current version
 * I have assumed that I have the methods pop() and push() defined in my CUDA kernel.
 * Therefore I just call them. In the later versions, I should probably implement
 * a "sequencer" that is a state machine and can support these operations and will
 * ultimately returns a C-code string representing the sequence of the operations that were
 * performed to evaluate this node. This can ultimately lead to a partially-CUDA implemented
 * ECJ.
 * 
 * Eg (code stub so that I can remember what I want to do in the future):
 *  Sequencer s = new M2XFilterSequencer();
 *  s.popAndStore();	// in C-Code will do: double first = pop(); 
 *  s.popAndStore();	// in C-Code will do: double second = pop();
 *  s.obtainForThisThread(M2XFilterSequencer.TARGET_SMALL_AVG); // in C-Code will do: double smallAvg = smallAvg[tid];
 *  s.do("here is some C code");
 *  s.finish();
 *  return s.getGeneratedCode();
 *  
 *  This whole thing seems a little bit far-fetched, but is absolutely possible.
 *  Whoever wants to use CUDA-ECJ must also implement the proper sequencer based
 *  on the kernel template that (s)he has developed. (I have always assumed "tid" to be the
 *  variable that gives me the mapped-index of the thread that is running this code)
 * 
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class CudaNode extends GPNode {
	
	private static final long serialVersionUID = -3678393067820352193L;
	
	/**
	 * The OPCODE of this class. CUDA interpreter uses this OPCODE
	 * to know the number of arguments of this node and also the type
	 * of the arguments.
	 * You should not modify this value and the CUDAInitializer will assign a
	 * generated value to this field.
	 * Note that each sub-class has a single OPCODE.
	 */
	protected byte opcode;
	
	/**
	 * Sets the OPCODE for the prototype of this class
	 * You should not call this function!
	 * 
	 * @param opcode
	 */
	public void setOpcode(byte opcode) {
		this.opcode = opcode;
	}
	
	/**
	 * Returns the OPCODE assosicated with this prototype.
	 * Will be later used to convert the tree to a RPN notation
	 * for transfer to CUDA memory. It was designed to return an array
	 * of bytes for those operators that can have constant values in front of 
	 * them (Eg. ERCs or random values)
	 *  
	 * @return
	 */
	public byte[] getOpcode() {
		return new byte[] {this.opcode};
	}
	
	/**
	 * Makes a postfix expression of this node and its child nodes.
	 * Note that this is a recursive function.
	 * 
	 * @return The byte string of the postfix-converted of this node and its child nodes
	 */
	public byte[] makePostfixExpression() {
		TByteList result = new TByteArrayList();
		// Add the terminating sequence! We will be reversing this guy later
		result.add((byte)0);
		
    	Stack<CudaNode> stack = new Stack<CudaNode>();
    	stack.push(this);
    	CudaNode node;
    	
    	while (!stack.isEmpty()) {
    		node = stack.pop();
    		// Get the opcodes and add them to this list
    		result.add(node.getOpcode());
    		
    		// Add the children of this node to the stack
    		for (int i = 0 ; i < node.children.length ; i++ ) {
    			stack.push((CudaNode) node.children[i]);
    		}
    	}
    	
    	// Reverse the thing
    	result.reverse();
    	return result.toArray();
	}
	
	@Override
	public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {
		// Do nothing!
		throw new RuntimeException("CPU Evaluation not implemented!");
	}
	
	/**
	 * Returns a C-like source code that will tell us the micro-operations
	 * that are required in order to evaluate this node.
	 * 
	 * @return		A C-like code, detailing the implementation of this tree
	 * 	node. (eg. float second = pop(); float first = pop(); push(first + second) )
	 */
	public abstract InstructionSequencer getCudaAction();
}
