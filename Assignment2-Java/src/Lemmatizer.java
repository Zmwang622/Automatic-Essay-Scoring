import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.TypesafeMap;


public class Lemmatizer {
	public static File f = new File("/Users/mingw/Downloads/bodies.csv");
	
	static String readFile(String path, Charset encoding) 
			  throws IOException 
	{
		  byte[] encoded = Files.readAllBytes(Paths.get(path));
		  return new String(encoded, encoding);
	}
	
	public static void main(String[] args) throws IOException, FileNotFoundException 
	{
		final String content = readFile("/Users/mingw/Downloads/bodies.csv",StandardCharsets.UTF_8);
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		CoreDocument document = new CoreDocument(content);
		pipeline.annotate(document);
		BufferedWriter writer = new 
				BufferedWriter(new FileWriter("C:/Users/mingw/FER/Assignment2-Java/Bodies_Lemmatized_Java.txt"));
		for(CoreLabel cl:document.tokens())
		{
			writer.write(cl.lemma());
			writer.newLine();
		}
		writer.close();
		System.out.println("Done");
	}
		
}