package searcher;

import java.io.*;
import java.util.*;

public class ContextSampler {
    String vocabFile;
    String dataFile;
    Set<String> vocab;

    static final int chunkSize = 8092;
    static byte[] buff = new byte[chunkSize];
    static final int K = 5;

    ContextSampler(String vocabFile, String dataFile) throws IOException {
        this.vocabFile = vocabFile;
        this.dataFile = dataFile;
        vocab = new HashSet<>();

        FileReader fr = new FileReader(vocabFile);
        BufferedReader br = new BufferedReader(fr);
        String line;

        while ((line = br.readLine()) != null) {
            vocab.add(line.trim());
        }

        br.close();
        fr.close();
    }

    public void process() throws Exception {
        FileWriter fw = new FileWriter("contexts.tsv");
        BufferedWriter bw = new BufferedWriter(fw);

        RandomAccessFile f = new RandomAccessFile(dataFile, "r");
        int offset = 0;
        int nbytesRead = 0;

        while (nbytesRead != -1) { 
            nbytesRead = f.read(buff);
            if (nbytesRead > 0)
                processChunk(buff, nbytesRead, bw);
        } 

        f.close();
        bw.close();
        fw.close();
    }

    void processChunk(byte[] byteArray, int n, BufferedWriter bw) throws Exception {
        String chunkText = new String(byteArray, 0, n, "UTF-8"); // for UTF-8 encoding
        //System.out.println(String.format("Chunk: |%s|", chunkText));    

        StringBuffer leftContextBuff = new StringBuffer();
        StringBuffer rightContextBuff = new StringBuffer();
        StringBuffer buff;
        String[] words = chunkText.split("\\s+");

        for (int i=0; i < words.length; i++) {
            if (vocab.contains(words[i])) { // word of interest
                leftContextBuff.setLength(0);
                rightContextBuff.setLength(0);

                for (int j=Math.max(0, i-K); j <= Math.min(i+K, words.length-1); j++) {
                    if (i==j)
                        continue;
                    buff = j < i? leftContextBuff: rightContextBuff;
                    buff.append(words[j]).append(" ");                    
                }
                bw.write(String.format("%s\t%s\t%s\n", words[i], leftContextBuff.toString(), rightContextBuff.toString()));    
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("usage: java ContextSampler <datafile> (document text) <vocab file (one word in each line)> OUT: writes a file 'contexts.tsv'");
            return;
        }    
       
        try { 
            ContextSampler sampler = new ContextSampler(args[1], args[0]);
            sampler.process();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
