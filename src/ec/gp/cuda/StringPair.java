package ec.gp.cuda;

import java.util.Map.Entry;

/**
 * Defines a pair of strings for storing multiple strings in a list
 * 
 * @author Mehran Maghoumi
 *
 */
public class StringPair implements Entry<String, String>{
	protected String key;
	protected String value;
	
	public StringPair(String key, String value) {
		this.key = key;
		this.value = value;
	}
	
	@Override
	public String getKey() {
		return key;
	}
	@Override
	public String getValue() {
		return value;
	}
	@Override
	public String setValue(String value) {
		return this.value = value;
	}
}
