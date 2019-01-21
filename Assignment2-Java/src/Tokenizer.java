import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;

public class Tokenizer {

	public static void main(String[] args) throws IOException, FileNotFoundException {
		// TODO Auto-generated method stub
		System.out.println("Hello");
		File f = new File("/Users/mingw/Downloads/bodies.csv");
		
		PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer<>(new FileReader(f), 
				new CoreLabelTokenFactory(),"");
		BufferedWriter writer = new 
				BufferedWriter(new FileWriter("C:/Users/mingw/FER/Assignment2-Java/Bodies_Tokenized_Java.txt"));
		
		
		while(ptbt.hasNext())
		{
			writer.write(ptbt.next().toString());
			writer.write("\n");
			//System.out.println(ptbt.next());
		}
		
		writer.close();
		System.out.print("It worked");
		
	}
		
}
